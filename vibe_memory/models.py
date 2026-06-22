"""Data models for the Vibe Memory system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Detail:
    """Raw message-level content."""
    detail_id: str
    role: str  # "user", "assistant", "tool", "system"
    content: str
    content_type: str  # "text", "code", "execution_output", etc.
    timestamp: Optional[datetime] = None


@dataclass
class Memory:
    """A meaningful chunk within a conversation."""
    memory_id: str
    text: str
    summary: str
    entities: list[str] = field(default_factory=list)
    emotion_tags: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    relevance_score: float = 1.0
    notable_quotes: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    """Individual chat session."""
    conversation_id: str
    title: str
    create_time: float  # epoch timestamp
    model_slug: Optional[str] = None
    message_count: int = 0
    summary: str = ""
    is_archived: bool = False


@dataclass
class Chapter:
    """Top-level era or period."""
    chapter_id: str
    title: str
    start_time: float  # epoch timestamp
    end_time: float  # epoch timestamp
    summary: str = ""
    themes: list[str] = field(default_factory=list)
    emotion_tags: list[str] = field(default_factory=list)


@dataclass
class MemoryExtraction:
    """Result of LLM extraction for a single memory chunk."""
    entities: list[str] = field(default_factory=list)
    emotion_tags: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    summary: str = ""
    notable_quotes: list[str] = field(default_factory=list)


@dataclass
class ConversationSummary:
    """Result of LLM summarization for a conversation."""
    title: str = ""
    summary: str = ""
    key_moments: list[str] = field(default_factory=list)
    emotional_arc: str = ""


@dataclass
class ChapterSummary:
    """Result of LLM summarization for a chapter/era."""
    title: str = ""
    summary: str = ""
    recurring_themes: list[str] = field(default_factory=list)
    character_notes: str = ""
