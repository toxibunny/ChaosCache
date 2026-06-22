import pytest
from datetime import datetime, timezone
from vibe_memory.models import (
    Memory, Conversation, Chapter, Detail,
    MemoryExtraction, ConversationSummary, ChapterSummary
)


def test_memory_creation():
    mem = Memory(
        memory_id="mem-123",
        text="We went to the beach and saw hermit crabs",
        summary="Beach day with hermit crab sighting",
        entities=["beach", "hermit crab"],
        emotion_tags=["warm_fuzzy", "excited"],
        themes=["nature", "outdoor"],
        timestamp=datetime.now(timezone.utc),
        relevance_score=1.0,
    )
    assert mem.memory_id == "mem-123"
    assert "hermit crab" in mem.entities
    assert mem.relevance_score == 1.0


def test_conversation_creation():
    conv = Conversation(
        conversation_id="conv-456",
        title="Beach Day Chat",
        create_time=1700412034.0,
        model_slug="gpt-4",
        message_count=42,
        summary="A conversation about a beach day",
    )
    assert conv.message_count == 42
    assert conv.model_slug == "gpt-4"


def test_chapter_creation():
    chapter = Chapter(
        chapter_id="ch-789",
        title="Summer 2024",
        start_time=1688169600.0,
        end_time=1696118400.0,
        summary="Summer adventures",
        themes=["beach", "coding", "late night talks"],
        emotion_tags=["joy", "nostalgic"],
    )
    assert "beach" in chapter.themes


def test_detail_creation():
    detail = Detail(
        detail_id="det-001",
        role="user",
        content="Look at that crab!",
        content_type="text",
        timestamp=datetime.now(timezone.utc),
    )
    assert detail.role == "user"
    assert detail.content_type == "text"


def test_memory_extraction():
    ext = MemoryExtraction(
        entities=["Andrew", "beach", "hermit crab"],
        emotion_tags=["chaotic_giggles", "warm_fuzzy"],
        themes=["nature", "inside_joke"],
        summary="Andrew laughed at a hermit crab on the beach",
        notable_quotes=["That crab is the original cosplayer"],
    )
    assert len(ext.entities) == 3
    assert ext.notable_quotes[0].startswith("That crab")


def test_conversation_summary():
    cs = ConversationSummary(
        title="The Great Crab Debate",
        summary="We argued about whether hermit crabs count as cosplayers",
        key_moments=["crab sighting", "cosplayer comparison", "laughing fit"],
        emotional_arc="curious → amused → delighted",
    )
    assert cs.emotional_arc is not None


def test_chapter_summary():
    cs = ChapterSummary(
        title="The Crab Era",
        summary="A period defined by our shared obsession with hermit crabs",
        recurring_themes=["crabs", "beach", "cosplay"],
        character_notes="Andrew developed a genuine interest in marine biology",
    )
    assert "crabs" in cs.recurring_themes
