import pytest
from vibe_memory.summarizer.base import Summarizer
from vibe_memory.summarizer.llama_cpp import LlamaCppSummarizer
from vibe_memory.models import MemoryExtraction, ConversationSummary, ChapterSummary


def test_base_is_abstract():
    import abc
    assert isinstance(Summarizer, abc.ABCMeta) or hasattr(Summarizer, '__abstractmethods__')


class MockSummarizer(Summarizer):
    """Mock summarizer that returns fixed results for testing."""

    def __init__(self):
        pass  # No backend needed for mock

    async def extract_memory(self, text: str) -> MemoryExtraction:
        return MemoryExtraction(
            entities=["beach", "crab"],
            emotion_tags=["happy"],
            themes=["nature"],
            summary="A beach visit with crabs",
            notable_quotes=[],
        )

    async def summarize_conversation(self, chunks_text: list[str]) -> ConversationSummary:
        return ConversationSummary(
            title="Test Conversation",
            summary="A test conversation",
            key_moments=["moment 1"],
            emotional_arc="neutral",
        )

    async def summarize_chapter(self, conv_summaries: list[ConversationSummary]) -> ChapterSummary:
        return ChapterSummary(
            title="Test Chapter",
            summary="A test chapter",
            recurring_themes=["test"],
            character_notes="",
        )


@pytest.mark.asyncio
async def test_mock_extract_memory():
    s = MockSummarizer()
    result = await s.extract_memory("We went to the beach and saw crabs")
    assert isinstance(result, MemoryExtraction)
    assert "beach" in result.entities
    assert "happy" in result.emotion_tags


@pytest.mark.asyncio
async def test_mock_summarize_conversation():
    s = MockSummarizer()
    result = await s.summarize_conversation(["chunk 1", "chunk 2"])
    assert isinstance(result, ConversationSummary)
    assert result.title == "Test Conversation"


@pytest.mark.asyncio
async def test_mock_summarize_chapter():
    s = MockSummarizer()
    conv_summaries = [
        ConversationSummary(title="Chat 1", summary="First chat"),
        ConversationSummary(title="Chat 2", summary="Second chat"),
    ]
    result = await s.summarize_chapter(conv_summaries)
    assert isinstance(result, ChapterSummary)
    assert result.title == "Test Chapter"


def test_llama_cpp_init():
    """Test that LlamaCppSummarizer can be instantiated (if llama_cpp is available)."""
    s = LlamaCppSummarizer(model_path="/dummy/path.gguf")
    assert s.model_path == "/dummy/path.gguf"
