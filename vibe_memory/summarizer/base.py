"""Base summarizer interface — pluggable LLM backends."""

from abc import ABC, abstractmethod
from vibe_memory.models import MemoryExtraction, ConversationSummary, ChapterSummary


class Summarizer(ABC):
    """Abstract base class for LLM-based summarization backends."""

    @abstractmethod
    async def extract_memory(self, text: str) -> MemoryExtraction:
        """Extract entities, emotions, themes, and summary from a text chunk.

        Args:
            text: Formatted text from a conversation chunk
                  (e.g., "[User] Hello\n\n[Assistant] Hi there!")

        Returns:
            MemoryExtraction with entities, emotion_tags, themes, summary, notable_quotes
        """

    @abstractmethod
    async def summarize_conversation(self, chunks_text: list[str]) -> ConversationSummary:
        """Generate a summary for an entire conversation.

        Args:
            chunks_text: List of formatted text blocks, one per chunk

        Returns:
            ConversationSummary with title, summary, key_moments, emotional_arc
        """

    @abstractmethod
    async def summarize_chapter(self, conv_summaries: list[ConversationSummary]) -> ChapterSummary:
        """Generate a summary for a chapter/era from conversation summaries.

        Args:
            conv_summaries: List of ConversationSummary objects for the chapter

        Returns:
            ChapterSummary with title, summary, recurring_themes, character_notes
        """
