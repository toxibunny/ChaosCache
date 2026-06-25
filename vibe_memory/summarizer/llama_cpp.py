"""Llama.cpp backend for local GGUF model inference."""

import json
import logging
from vibe_memory.summarizer.base import Summarizer
from vibe_memory.models import MemoryExtraction, ConversationSummary, ChapterSummary

logger = logging.getLogger(__name__)

# Prompts for structured extraction
MEMORY_EXTRACTION_PROMPT = """\
Analyze the conversation chunk below and extract key information.
Return ONLY valid JSON with this structure:

{{
  "entities": ["person", "place", "object"],
  "emotion_tags": ["emotion1", "emotion2"],
  "themes": ["theme1", "theme2"],
  "summary": "Brief one-sentence summary of the actual conversation content.",
  "notable_quotes": ["Quote from the conversation"]
}}

Conversation chunk to analyze:
{text}

Output JSON:"""

CONVERSATION_SUMMARY_PROMPT = """\
Summarize the conversation below as JSON. Return ONLY valid JSON.

{{
  "title": "A catchy, specific title based on the actual conversation topic",
  "summary": "A paragraph summarizing the key points discussed.",
  "key_moments": ["First key moment", "Second key moment", "Third key moment"],
  "emotional_arc": "How the mood shifted, e.g. 'curious -> amused -> thoughtful'"
}}

Conversation chunks to summarize:
{chunks}

Output JSON:"""

CHAPTER_SUMMARY_PROMPT = """\
Summarize this era/chapter of conversations as JSON. Return ONLY valid JSON.

{{
  "title": "A title that captures the theme of this time period",
  "summary": "A narrative summary of what happened during this period.",
  "recurring_themes": ["theme1", "theme2", "theme3"],
  "character_notes": "Notes about how the people/relationship evolved over this time."
}}

Conversation summaries from this period:
{summaries}

Output JSON:"""


class LlamaCppSummarizer(Summarizer):
    """Summarizer using llama-cpp-python for local GGUF model inference."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: int = 8,
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.max_retries = max_retries
        self._llama = None

    def _get_llama(self):
        """Lazy load llama_cpp to avoid import errors if not installed."""
        from llama_cpp import Llama
        if self._llama is None:
            self._llama = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
        return self._llama

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
        llama = self._get_llama()
        for attempt in range(self.max_retries):
            try:
                output = llama(
                    prompt,
                    max_tokens=1024,
                    temperature=self.temperature,
                    stop=["}]"],
                    echo=False,
                )
                return output["choices"][0]["text"].strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
        return ""

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        return json.loads(text)
