import pytest
from vibe_memory.graph.builder import GraphBuilder
from vibe_memory.models import (
    Memory, Conversation, Chapter, Detail,
    MemoryExtraction, ConversationSummary, ChapterSummary,
)
from datetime import datetime, timezone


class MockDriver:
    """Mock Neo4j driver for testing without a real database."""

    def __init__(self):
        self.nodes = []
        self.relationships = []
        self.schema_commands = []

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def run(self, query, parameters=None):
        self.schema_commands.append((query, parameters or {}))
        # Track nodes created
        if "CREATE" in query and "(:" in query:
            for node_type in ["Chapter", "Conversation", "Memory", "Detail", "Entity", "Emotion"]:
                if node_type in query:
                    self.nodes.append((node_type, parameters))
        if "CREATE" in query and "-[" in query:
            for rel_type in ["CONTAINS", "HAS_DETAIL", "RELATES_TO", "HAS_EMOTION", "MENTIONS", "APPEARS_WITH", "SARAH_APPROVED"]:
                if rel_type in query:
                    self.relationships.append((rel_type, parameters))
        return {"single": lambda: {"value": len(self.nodes)}}


def test_create_schema():
    driver = MockDriver()
    builder = GraphBuilder(driver)
    builder.create_schema()
    # Should have created constraints and indexes
    assert len(driver.schema_commands) > 0


def test_add_chapter():
    driver = MockDriver()
    builder = GraphBuilder(driver)
    chapter = Chapter(
        chapter_id="ch-1",
        title="Summer 2024",
        start_time=1688169600.0,
        end_time=1696118400.0,
        summary="Summer adventures",
        themes=["beach", "coding"],
        emotion_tags=["joy"],
    )
    builder.add_chapter(chapter)
    assert any("Chapter" in str(c) for c in driver.schema_commands)


def test_add_conversation():
    driver = MockDriver()
    builder = GraphBuilder(driver)
    conv = Conversation(
        conversation_id="conv-1",
        title="Beach Day",
        create_time=1700412034.0,
        model_slug="gpt-4",
        message_count=42,
        summary="We went to the beach",
    )
    builder.add_conversation(conv, chapter_id="ch-1")
    assert any("Conversation" in str(c) for c in driver.schema_commands)


def test_add_memory_with_entities():
    driver = MockDriver()
    builder = GraphBuilder(driver)
    mem = Memory(
        memory_id="mem-1",
        text="We saw hermit crabs on the beach",
        summary="Hermit crab sighting",
        entities=["hermit crab", "beach"],
        emotion_tags=["excited", "warm_fuzzy"],
        themes=["nature"],
        timestamp=datetime.now(timezone.utc),
        relevance_score=1.0,
    )
    builder.add_memory(mem, conversation_id="conv-1")
    # Should create Memory node + Entity nodes + Emotion nodes + relationships
    assert any("Memory" in str(c) for c in driver.schema_commands)
    assert any("Entity" in str(c) for c in driver.schema_commands)
    assert any("Emotion" in str(c) for c in driver.schema_commands)


def test_add_relationships():
    driver = MockDriver()
    builder = GraphBuilder(driver)
    # Add two memories and create a relationship
    mem1 = Memory(memory_id="mem-1", text="First memory", summary="First")
    mem2 = Memory(memory_id="mem-2", text="Second memory", summary="Second")
    builder.add_memory(mem1, conversation_id="conv-1")
    builder.add_memory(mem2, conversation_id="conv-1")
    builder.add_relationships("mem-1", "mem-2", relationship_type="RELATES_TO", weight=0.8)
    assert any("RELATES_TO" in str(c) for c in driver.schema_commands)
