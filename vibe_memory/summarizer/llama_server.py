"""Llama.cpp server backend for remote GGUF model inference via HTTP API."""

import json
import logging
from vibe_memory.summarizer.base import Summarizer
from vibe_memory.models import MemoryExtraction, ConversationSummary, ChapterSummary

logger = logging.getLogger(__name__)

# Reuse prompts from the llama_cpp backend
from vibe_memory.summarizer.llama_cpp import (
    MEMORY_EXTRACTION_PROMPT,
    CONVERSATION_SUMMARY_PROMPT,
    CHAPTER_SUMMARY_PROMPT,
)


class LlamaServerSummarizer(Summarizer):
    """Summarizer using llama.cpp server API (HTTP) for remote model inference."""

    def __init__(
        self,
        server_url: str = "http://localhost:8081",
        model_name: str = "",
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        import httpx
        self._client = httpx.Client(base_url=self.server_url, timeout=120.0)

    async def extract_memory(self, text: str) -> MemoryExtraction:
        prompt = MEMORY_EXTRACTION_PROMPT.format(text=text)
        response = await self._call_llm(prompt)
        data = self._parse_json(response)
        return MemoryExtraction(
            entities=data.get("entities", []),
            emotion_tags=data.get("emotion_tags", []),
            themes=data.get("themes", []),
            summary=data.get("summary", ""),
            notable_quotes=data.get("notable_quotes", []),
        )

    async def summarize_conversation(self, chunks_text: list[str]) -> ConversationSummary:
        chunks_str = "\n\n---\n\n".join(chunks_text)
        prompt = CONVERSATION_SUMMARY_PROMPT.format(chunks=chunks_str)
        response = await self._call_llm(prompt)
        data = self._parse_json(response)
        return ConversationSummary(
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            key_moments=data.get("key_moments", []),
            emotional_arc=data.get("emotional_arc", ""),
        )

    async def summarize_chapter(self, conv_summaries: list[ConversationSummary]) -> ChapterSummary:
        summaries_str = "\n\n".join(
            f"- {cs.title}: {cs.summary}" for cs in conv_summaries
        )
        prompt = CHAPTER_SUMMARY_PROMPT.format(summaries=summaries_str)
        response = await self._call_llm(prompt)
        data = self._parse_json(response)
        return ChapterSummary(
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            recurring_themes=data.get("recurring_themes", []),
            character_notes=data.get("character_notes", ""),
        )

    async def _call_llm(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                body = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1024,
                    "temperature": self.temperature,
                }
                if self.model_name:
                    body["model"] = self.model_name
                response = self._client.post("/v1/chat/completions", json=body)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
        return ""

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        return json.loads(text)
