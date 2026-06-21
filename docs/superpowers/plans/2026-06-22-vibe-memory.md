# Vibe Memory System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python library that ingests chat logs, summarizes them with a local LLM, stores them in a Neo4j graph, and retrieves vibe-matched memories with configurable serendipity.

**Architecture:** Four-component pipeline (Ingestor → Summarizer → Graph Builder → Retrieval Engine) with pluggable LLM backends. Neo4j for graph storage, SQLite for source data.

**Tech Stack:** Python 3.10+, Neo4j (bolt driver), llama-cpp-python, sqlite3 (built-in)

## Global Constraints

- Python 3.10+ (type hints, match statements optional)
- Neo4j 5.x compatible driver (`neo4x` package)
- Local LLM via llama-cpp-python (GGUF models)
- All async summarizer methods (LLM calls are I/O bound)
- Package name: `vibe_memory`
- Tests use `pytest` + `pytest-asyncio`
- Non-fatal error handling: skip and log, never halt the pipeline
- Resumable ingestion: skip conversations already in Neo4j

---

### Task 1: Project Scaffolding + Data Models

**Files:**
- Create: `vibe_memory/__init__.py`
- Create: `vibe_memory/models.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`
- Create: `pyproject.toml`
- Create: `vibe_memory/summarizer/__init__.py`
- Create: `vibe_memory/graph/__init__.py`
- Create: `vibe_memory/retrieval/__init__.py`
- Create: `vibe_memory/chaos/__init__.py`
- Create: `vibe_memory/dream/__init__.py`

**Interfaces:**
- Produces: Data classes `Memory`, `Conversation`, `Chapter`, `Detail`, `MemoryExtraction`, `ConversationSummary`, `ChapterSummary`
- Produces: `pyproject.toml` with dependencies

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/toxi/c/archive && python -m pytest tests/test_models.py -v`
Expected: FAIL — module `vibe_memory` not found

- [ ] **Step 3: Write minimal implementation**

Create `vibe_memory/__init__.py`:

```python
"""Vibe Memory — memory graph for AI companion bots."""

from vibe_memory.models import (
    Memory,
    Conversation,
    Chapter,
    Detail,
    MemoryExtraction,
    ConversationSummary,
    ChapterSummary,
)

__all__ = [
    "Memory",
    "Conversation",
    "Chapter",
    "Detail",
    "MemoryExtraction",
    "ConversationSummary",
    "ChapterSummary",
]
```

Create `vibe_memory/models.py`:

```python
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
```

Create empty `__init__.py` files for subpackages:

```python
# vibe_memory/summarizer/__init__.py
# vibe_memory/graph/__init__.py
# vibe_memory/retrieval/__init__.py
# vibe_memory/chaos/__init__.py
# vibe_memory/dream/__init__.py
# tests/__init__.py
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "vibe-memory"
version = "0.1.0"
description = "Memory graph for AI companion bots — vibes, emotions, inside jokes"
requires-python = ">=3.10"
dependencies = [
    "neo4x>=5.0",
    "llama-cpp-python>=0.2.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.14",
]
ollama = ["ollama>=0.1"]
openai = ["openai>=1.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/toxi/c/archive && pip install -e ".[dev]" && python -m pytest tests/test_models.py -v`
Expected: PASS — all 7 tests pass

- [ ] **Step 5: Commit**

```bash
git add vibe_memory/ tests/ pyproject.toml
git commit -m "feat: project scaffolding and data models

- Data classes: Memory, Conversation, Chapter, Detail
- Extraction/summary result types
- Package structure with summarizer/, graph/, retrieval/, chaos/, dream/
- pyproject.toml with dependencies"
```

---

### Task 2: Ingestor (chats.db → Chunks)

**Files:**
- Create: `vibe_memory/ingestor.py`
- Create: `tests/test_ingestor.py`

**Interfaces:**
- Consumes: `Conversation`, `Detail` from models
- Produces: `Ingestor` class with `load_conversations()`, `chunk_conversation()`, `iterate_chunks()`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestor.py`:

```python
import sqlite3
import pytest
from vibe_memory.ingestor import Ingestor
from vibe_memory.models import Conversation, Detail


def _create_test_db():
    """Create a minimal chats.db in memory for testing."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            title TEXT,
            create_time REAL,
            model_slug TEXT,
            message_count INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            content_type TEXT DEFAULT 'text',
            content TEXT,
            create_time REAL
        );
    """)
    conn.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                 ("conv-1", "Test Chat", 1700412034.0, "gpt-4", 4))
    conn.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                 ("msg-1", "conv-1", 0, "user", "text", "Hi there!", 1700412034.0))
    conn.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                 ("msg-2", "conv-1", 1, "assistant", "text", "Hello! How can I help?", 1700412035.0))
    conn.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                 ("msg-3", "conv-1", 2, "user", "text", "Tell me about crabs", 1700412036.0))
    conn.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                 ("msg-4", "conv-1", 3, "assistant", "text", "Crabs are cool!", 1700412037.0))
    return conn


def test_load_conversations():
    conn = _create_test_db()
    ing = Ingestor(conn)
    convos = ing.load_conversations()
    assert len(convos) == 1
    assert convos[0].conversation_id == "conv-1"
    assert convos[0].title == "Test Chat"
    assert convos[0].message_count == 4


def test_chunk_conversation():
    conn = _create_test_db()
    ing = Ingestor(conn)
    convos = ing.load_conversations()
    chunks = ing.chunk_conversation(convos[0])
    # Default chunk size groups user+assistant turns
    assert len(chunks) >= 1
    # Each chunk has messages in order
    for chunk in chunks:
        assert len(chunk) >= 1
        for msg in chunk:
            assert msg.role in ("user", "assistant", "tool")


def test_iterate_chunks():
    conn = _create_test_db()
    ing = Ingestor(conn)
    all_chunks = list(ing.iterate_chunks())
    assert len(all_chunks) > 0
    # Each item is (conversation, chunk_index, messages)
    for conv, chunk_idx, messages in all_chunks:
        assert isinstance(conv, Conversation)
        assert isinstance(chunk_idx, int)
        assert len(messages) > 0


def test_chunk_size_parameter():
    conn = _create_test_db()
    ing = Ingestor(conn, chunk_size=2)
    convos = ing.load_conversations()
    chunks = ing.chunk_conversation(convos[0])
    # With chunk_size=2, each chunk has at most 2 messages
    for chunk in chunks:
        assert len(chunk) <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestor.py -v`
Expected: FAIL — module `vibe_memory.ingestor` not found

- [ ] **Step 3: Write minimal implementation**

Create `vibe_memory/ingestor.py`:

```python
"""Ingestor — reads chats.db and chunks conversations for summarization."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

from vibe_memory.models import Conversation, Detail


@dataclass
class Message:
    """A single message from the database."""
    message_id: str
    conversation_id: str
    message_index: int
    role: str
    content_type: str
    content: str
    create_time: Optional[float] = None


class Ingestor:
    """Reads conversations and messages from chats.db and chunks them."""

    def __init__(self, db_path_or_conn, chunk_size: int = 10):
        """
        Args:
            db_path_or_conn: Path to chats.db or an existing sqlite3 connection.
            chunk_size: Maximum messages per chunk. Default 10 (5 turn pairs).
        """
        if isinstance(db_path_or_conn, str):
            self._conn = sqlite3.connect(db_path_or_conn)
            self._owns_conn = True
        else:
            self._conn = db_path_or_conn
            self._owns_conn = False
        self.chunk_size = chunk_size

    def close(self):
        if self._owns_conn:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def load_conversations(self) -> list[Conversation]:
        """Load all conversations from the database."""
        cursor = self._conn.execute("""
            SELECT conversation_id, title, create_time, model_slug,
                   message_count, is_archived
            FROM conversations
            ORDER BY create_time
        """)
        results = []
        for row in cursor.fetchall():
            results.append(Conversation(
                conversation_id=row[0],
                title=row[1] or "",
                create_time=row[2] or 0.0,
                model_slug=row[3],
                message_count=row[4] or 0,
                is_archived=bool(row[5]) if row[5] is not None else False,
            ))
        return results

    def _load_messages(self, conversation_id: str) -> list[Message]:
        """Load all messages for a conversation, ordered by message_index."""
        cursor = self._conn.execute("""
            SELECT message_id, conversation_id, message_index, role,
                   content_type, content, create_time
            FROM messages
            WHERE conversation_id = ?
            ORDER BY message_index
        """, (conversation_id,))
        results = []
        for row in cursor.fetchall():
            results.append(Message(
                message_id=row[0],
                conversation_id=row[1],
                message_index=row[2],
                role=row[3],
                content_type=row[4] or "text",
                content=row[5] or "",
                create_time=row[6],
            ))
        return messages

    def chunk_conversation(self, conversation: Conversation) -> list[list[Message]]:
        """Split a conversation's messages into chunks of chunk_size."""
        messages = self._load_messages(conversation.conversation_id)
        # Filter out system messages for chunking
        visible = [m for m in messages if m.role in ("user", "assistant", "tool")]
        chunks = []
        for i in range(0, len(visible), self.chunk_size):
            chunks.append(visible[i:i + self.chunk_size])
        return chunks

    def iterate_chunks(self) -> Iterator[tuple[Conversation, int, list[Message]]]:
        """Iterate over all conversations and their chunks.

        Yields:
            (conversation, chunk_index, messages_in_chunk)
        """
        conversations = self.load_conversations()
        for conv in conversations:
            chunks = self.chunk_conversation(conv)
            for idx, chunk in enumerate(chunks):
                yield conv, idx, chunk

    def _messages_to_text(self, messages: list[Message]) -> str:
        """Convert a list of messages to a text block for summarization."""
        parts = []
        for msg in messages:
            role_label = msg.role.capitalize()
            parts.append(f"[{role_label}] {msg.content}")
        return "\n\n".join(parts)
```

- [ ] **Step 4: Fix bug — variable name mismatch**

The `_load_messages` method assigns to `results` but returns `messages`. Fix:

```python
    def _load_messages(self, conversation_id: str) -> list[Message]:
        """Load all messages for a conversation, ordered by message_index."""
        cursor = self._conn.execute("""
            SELECT message_id, conversation_id, message_index, role,
                   content_type, content, create_time
            FROM messages
            WHERE conversation_id = ?
            ORDER BY message_index
        """, (conversation_id,))
        results = []
        for row in cursor.fetchall():
            results.append(Message(
                message_id=row[0],
                conversation_id=row[1],
                message_index=row[2],
                role=row[3],
                content_type=row[4] or "text",
                content=row[5] or "",
                create_time=row[6],
            ))
        return results
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestor.py -v`
Expected: PASS — all 4 tests pass

- [ ] **Step 6: Commit**

```bash
git add vibe_memory/ingestor.py tests/test_ingestor.py
git commit -m "feat: ingestor — reads chats.db and chunks conversations

- Ingestor class with load_conversations(), chunk_conversation(), iterate_chunks()
- Configurable chunk_size for controlling summarization granularity
- Context manager support for connection handling
- Filters system messages, preserves user/assistant/tool roles"
```

---

### Task 3: Summarizer Base + Llama-CPP Backend

**Files:**
- Create: `vibe_memory/summarizer/base.py` (move `__init__.py` content here)
- Modify: `vibe_memory/summarizer/__init__.py`
- Create: `vibe_memory/summarizer/llama_cpp.py`
- Create: `tests/test_summarizer.py`

**Interfaces:**
- Consumes: `MemoryExtraction`, `ConversationSummary`, `ChapterSummary` from models
- Consumes: Text chunks from Ingestor
- Produces: `Summarizer` abstract base class with `extract_memory()`, `summarize_conversation()`, `summarize_chapter()`
- Produces: `LlamaCppSummarizer` concrete implementation

- [ ] **Step 1: Write the failing test**

Create `tests/test_summarizer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_summarizer.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Write minimal implementation**

Create `vibe_memory/summarizer/base.py`:

```python
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
```

Create `vibe_memory/summarizer/llama_cpp.py`:

```python
"""Llama.cpp backend for local GGUF model inference."""

import json
import logging
from vibe_memory.summarizer.base import Summarizer
from vibe_memory.models import MemoryExtraction, ConversationSummary, ChapterSummary

logger = logging.getLogger(__name__)

# Prompts for structured extraction
MEMORY_EXTRACTION_PROMPT = """\
Analyze this conversation chunk and extract the following as JSON:
{
  "entities": ["list of people, places, objects, inside jokes mentioned"],
  "emotion_tags": ["list of emotions present, e.g. warm_fuzzy, chaotic_giggles, nostalgic"],
  "themes": ["list of themes, e.g. beach day, coding session, late night talk"],
  "summary": "one sentence summary of what happened",
  "notable_quotes": ["any funny or memorable lines"]
}

Conversation chunk:
{text}
"""

CONVERSATION_SUMMARY_PROMPT = """\
Summarize this conversation as JSON:
{
  "title": "a catchy title for this conversation",
  "summary": "a paragraph summarizing the key points",
  "key_moments": ["list of 3-5 key moments"],
  "emotional_arc": "how the mood shifted, e.g. 'curious -> amused -> delighted'"
}

Conversation chunks:
{chunks}
"""

CHAPTER_SUMMARY_PROMPT = """\
Summarize this era/chapter of conversations as JSON:
{
  "title": "a title for this era",
  "summary": "a narrative summary of this period",
  "recurring_themes": ["themes that appeared across multiple conversations"],
  "character_notes": "notes about how the people/relationship evolved"
}

Conversation summaries:
{summaries}
"""


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
        lazy load llama_cpp to avoid import errors if not installed
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
```

Modify `vibe_memory/summarizer/__init__.py`:

```python
from vibe_memory.summarizer.base import Summarizer
from vibe_memory.summarizer.llama_cpp import LlamaCppSummarizer

__all__ = ["Summarizer", "LlamaCppSummarizer"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_summarizer.py -v`
Expected: PASS — all 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add vibe_memory/summarizer/ tests/test_summarizer.py
git commit -m "feat: summarizer base class and llama-cpp backend

- Abstract Summarizer with extract_memory, summarize_conversation, summarize_chapter
- LlamaCppSummarizer for local GGUF models
- Structured JSON prompts for entity/emotion/theme extraction
- MockSummarizer in tests for unit testing without LLM
- Retry logic and JSON parsing with markdown handling"
```

---

### Task 4: Graph Builder (Neo4j Schema + Node Creation)

**Files:**
- Create: `vibe_memory/graph/schema.py`
- Create: `vibe_memory/graph/builder.py`
- Modify: `vibe_memory/graph/__init__.py`
- Create: `tests/test_graph.py`

**Interfaces:**
- Consumes: `Memory`, `Conversation`, `Chapter`, `Detail` from models
- Consumes: `MemoryExtraction`, `ConversationSummary`, `ChapterSummary` from models
- Produces: `GraphBuilder` class with `create_schema()`, `add_chapter()`, `add_conversation()`, `add_memory()`, `add_relationships()`

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph.py`:

```python
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
    assert len(builder.schema_commands) > 0


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
    assert any("Chapter" in str(c) for c in builder.schema_commands)


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
    assert any("Conversation" in str(c) for c in builder.schema_commands)


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
    assert any("Memory" in str(c) for c in builder.schema_commands)
    assert any("Entity" in str(c) for c in builder.schema_commands)
    assert any("Emotion" in str(c) for c in builder.schema_commands)


def test_add_relationships():
    driver = MockDriver()
    builder = GraphBuilder(driver)
    # Add two memories and create a relationship
    mem1 = Memory(memory_id="mem-1", text="First memory", summary="First")
    mem2 = Memory(memory_id="mem-2", text="Second memory", summary="Second")
    builder.add_memory(mem1, conversation_id="conv-1")
    builder.add_memory(mem2, conversation_id="conv-1")
    builder.add_relationships("mem-1", "mem-2", relationship_type="RELATES_TO", weight=0.8)
    assert any("RELATES_TO" in str(c) for c in builder.schema_commands)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -v`
Expected: FAIL — module `vibe_memory.graph.builder` not found

- [ ] **Step 3: Write minimal implementation**

Create `vibe_memory/graph/schema.py`:

```python
"""Neo4j schema definitions for the Vibe Memory graph."""

# Node labels
LABEL_CHAPTER = "Chapter"
LABEL_CONVERSATION = "Conversation"
LABEL_MEMORY = "Memory"
LABEL_DETAIL = "Detail"
LABEL_ENTITY = "Entity"
LABEL_EMOTION = "Emotion"

# Relationship types
REL_CONTAINS = "CONTAINS"           # Chapter→Conversation, Conversation→Memory
REL_HAS_DETAIL = "HAS_DETAIL"       # Memory→Detail
REL_RELATES_TO = "RELATES_TO"       # Memory→Memory (cross-links)
REL_HAS_EMOTION = "HAS_EMOTION"     # Memory→Emotion
REL_MENTIONS = "MENTIONS"           # Memory→Entity
REL_APPEARS_WITH = "APPEARS_WITH"   # Entity→Entity (co-occurrence)
REL_SARAH_APPROVED = "SARAH_APPROVED"  # Memory→Memory (memories that slap)

# Schema creation Cypher
CREATE_CONSTRAINTS = """
CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE;
CREATE CONSTRAINT conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.conversation_id IS UNIQUE;
CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE;
CREATE CONSTRAINT detail_id IF NOT EXISTS FOR (d:Detail) REQUIRE d.detail_id IS UNIQUE;
CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;
CREATE CONSTRAINT emotion_name IF NOT EXISTS FOR (em:Emotion) REQUIRE em.name IS UNIQUE;

CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp);
CREATE INDEX memory_emotion_tags IF NOT EXISTS FOR (m:Memory) ON (m.emotion_tags);
CREATE INDEX memory_themes IF NOT EXISTS FOR (m:Memory) ON (m.themes);
CREATE INDEX memory_relevance IF NOT EXISTS FOR (m:Memory) ON (m.relevance_score);
CREATE INDEX entity_name_lookup IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX emotion_name_lookup IF NOT EXISTS FOR (em:Emotion) ON (em.name);
CREATE INDEX conversation_time IF NOT EXISTS FOR (c:Conversation) ON (c.create_time);
"""
```

Create `vibe_memory/graph/builder.py`:

```python
"""Graph Builder — creates Neo4j nodes and relationships from summarized data."""

import logging
from datetime import datetime, timezone
from typing import Optional

from vibe_memory.graph import schema
from vibe_memory.models import (
    Memory, Conversation, Chapter, Detail,
    MemoryExtraction, ConversationSummary, ChapterSummary,
)

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds the Neo4j graph from summarized conversation data."""

    def __init__(self, driver):
        """
        Args:
            driver: Neo4j driver instance (from neo4x.GraphDatabase.driver)
        """
        self.driver = driver

    def create_schema(self):
        """Create constraints and indexes in Neo4j."""
        with self.driver.session() as session:
            session.run(schema.CREATE_CONSTRAINTS)
        logger.info("Schema created/verified")

    def add_chapter(self, chapter: Chapter):
        """Add a Chapter node to the graph."""
        with self.driver.session() as session:
            session.run(f"""
                MERGE (c:{schema.LABEL_CHAPTER} {{chapter_id: $chapter_id}})
                SET c += $properties
            """, {
                "chapter_id": chapter.chapter_id,
                "properties": {
                    "title": chapter.title,
                    "start_time": chapter.start_time,
                    "end_time": chapter.end_time,
                    "summary": chapter.summary,
                    "themes": chapter.themes,
                    "emotion_tags": chapter.emotion_tags,
                }
            })
        logger.info(f"Added chapter: {chapter.title}")

    def add_conversation(self, conversation: Conversation, chapter_id: Optional[str] = None):
        """Add a Conversation node and link to chapter if provided."""
        with self.driver.session() as session:
            session.run(f"""
                MERGE (c:{schema.LABEL_CONVERSATION} {{conversation_id: $conv_id}})
                SET c += $properties
            """, {
                "conv_id": conversation.conversation_id,
                "properties": {
                    "title": conversation.title,
                    "create_time": conversation.create_time,
                    "model_slug": conversation.model_slug,
                    "message_count": conversation.message_count,
                    "summary": conversation.summary,
                    "is_archived": conversation.is_archived,
                }
            })
            if chapter_id:
                session.run(f"""
                    MATCH (ch:{schema.LABEL_CHAPTER} {{chapter_id: $chapter_id}})
                    MATCH (c:{schema.LABEL_CONVERSATION} {{conversation_id: $conv_id}})
                    MERGE (ch)-[:{schema.REL_CONTAINS}]->(c)
                """, {
                    "chapter_id": chapter_id,
                    "conv_id": conversation.conversation_id,
                })
        logger.info(f"Added conversation: {conversation.title}")

    def add_memory(self, memory: Memory, conversation_id: str):
        """Add a Memory node with entity and emotion relationships."""
        with self.driver.session() as session:
            # Create Memory node
            ts = memory.timestamp.timestamp() if memory.timestamp else None
            session.run(f"""
                MERGE (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                SET m += $properties
            """, {
                "mem_id": memory.memory_id,
                "properties": {
                    "text": memory.text,
                    "summary": memory.summary,
                    "entities": memory.entities,
                    "emotion_tags": memory.emotion_tags,
                    "themes": memory.themes,
                    "timestamp": ts,
                    "relevance_score": memory.relevance_score,
                    "notable_quotes": memory.notable_quotes or [],
                }
            })
            # Link to conversation
            session.run(f"""
                MATCH (c:{schema.LABEL_CONVERSATION} {{conversation_id: $conv_id}})
                MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                MERGE (c)-[:{schema.REL_CONTAINS}]->(m)
            """, {
                "conv_id": conversation_id,
                "mem_id": memory.memory_id,
            })
            # Create Entity nodes and relationships
            for entity_name in memory.entities:
                session.run(f"""
                    MERGE (e:{schema.LABEL_ENTITY} {{name: $name}})
                    WITH e
                    MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                    MERGE (m)-[:{schema.REL_MENTIONS}]->(e)
                """, {"name": entity_name, "mem_id": memory.memory_id})
            # Create Emotion nodes and relationships
            for emotion_name in memory.emotion_tags:
                session.run(f"""
                    MERGE (em:{schema.LABEL_EMOTION} {{name: $name}})
                    WITH em
                    MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                    MERGE (m)-[:{schema.REL_HAS_EMOTION}]->(em)
                """, {"name": emotion_name, "mem_id": memory.memory_id})
            # Create Entity co-occurrence relationships
            for i, e1 in enumerate(memory.entities):
                for e2 in memory.entities[i+1:]:
                    session.run(f"""
                        MATCH (a:{schema.LABEL_ENTITY} {{name: $name1}})
                        MATCH (b:{schema.LABEL_ENTITY} {{name: $name2}})
                        MERGE (a)-[:{schema.REL_APPEARS_WITH}]->(b)
                    """, {"name1": e1, "name2": e2})
        logger.info(f"Added memory: {memory.summary[:50]}")

    def add_detail(self, detail: Detail, memory_id: str):
        """Add a Detail node linked to a Memory."""
        with self.driver.session() as session:
            ts = detail.timestamp.timestamp() if detail.timestamp else None
            session.run(f"""
                MERGE (d:{schema.LABEL_DETAIL} {{detail_id: $det_id}})
                SET d.role = $role, d.content = $content,
                    d.content_type = $content_type, d.timestamp = $timestamp
                WITH d
                MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                MERGE (m)-[:{schema.REL_HAS_DETAIL}]->(d)
            """, {
                "det_id": detail.detail_id,
                "role": detail.role,
                "content": detail.content,
                "content_type": detail.content_type,
                "timestamp": ts,
                "mem_id": memory_id,
            })

    def add_relationships(self, memory_id_1: str, memory_id_2: str,
                          relationship_type: str = "RELATES_TO", weight: float = 0.5):
        """Add a relationship between two Memory nodes."""
        rel = relationship_type if relationship_type in (
            schema.REL_RELATES_TO, schema.REL_SARAH_APPROVED
        ) else schema.REL_RELATES_TO
        with self.driver.session() as session:
            session.run(f"""
                MATCH (a:{schema.LABEL_MEMORY} {{memory_id: $mem_id_1}})
                MATCH (b:{schema.LABEL_MEMORY} {{memory_id: $mem_id_2}})
                MERGE (a)-[r:{rel}]->(b)
                SET r.weight = $weight
            """, {
                "mem_id_1": memory_id_1,
                "mem_id_2": memory_id_2,
                "weight": weight,
            })

    def update_relevance_score(self, memory_id: str, delta: float):
        """Update a memory's relevance score (for decay/revival)."""
        with self.driver.session() as session:
            session.run(f"""
                MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                SET m.relevance_score = COALESCE(m.relevance_score, 1.0) + $delta
                SET m.relevance_score = MIN(MAX(m.relevance_score, 0.0), 2.0)
            """, {"mem_id": memory_id, "delta": delta})

    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()
```

Modify `vibe_memory/graph/__init__.py`:

```python
from vibe_memory.graph.builder import GraphBuilder
from vibe_memory.graph import schema

__all__ = ["GraphBuilder", "schema"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph.py -v`
Expected: PASS — all 6 tests pass

- [ ] **Step 5: Commit**

```bash
git add vibe_memory/graph/ tests/test_graph.py
pip install neo4x
git commit -m "feat: graph builder — Neo4j schema and node creation

- Schema: constraints, indexes for all node types
- GraphBuilder: add_chapter, add_conversation, add_memory, add_detail
- Auto-creates Entity and Emotion nodes with relationships
- Entity co-occurrence (APPEARS_WITH) edges
- Relevance score updates for decay/revival
- SARAH_APPROVED edge type support"
```

---

### Task 5: Retrieval Engine (Context + Structured Queries)

**Files:**
- Create: `vibe_memory/retrieval/engine.py`
- Modify: `vibe_memory/retrieval/__init__.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Memory`, `Conversation`, `Chapter` from models
- Consumes: Neo4j graph from GraphBuilder
- Produces: `RetrievalEngine` with `retrieve()`, `query()`, `search_text()`, `reminds_me_of()`
- Produces: `MemoryStore` high-level class (the main public API)

- [ ] **Step 1: Write the failing test**

Create `tests/test_retrieval.py`:

```python
import pytest
from vibe_memory.retrieval.engine import RetrievalEngine, MemoryStore


class MockSession:
    """Mock Neo4j session for testing."""

    def __init__(self, mock_data=None):
        self.mock_data = mock_data or []
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def run(self, query, parameters=None):
        self.queries.append((query, parameters or {}))
        return MockResult(self.mock_data)


class MockResult:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for item in self.data:
            yield {"data": item}


class MockDriver:
    def __init__(self, mock_data=None):
        self.mock_data = mock_data

    def session(self):
        return MockSession(self.mock_data)


def test_retrieve_by_entities():
    mock_data = [
        {"memory_id": "mem-1", "text": "We went to the beach",
         "summary": "Beach day", "relevance_score": 1.0,
         "emotion_tags": ["happy"], "entities": ["beach"],
         "themes": [], "timestamp": 1700412034.0, "notable_quotes": []},
        {"memory_id": "mem-2", "text": "Saw crabs at the beach",
         "summary": "Crab sighting", "relevance_score": 0.8,
         "emotion_tags": ["excited"], "entities": ["crab", "beach"],
         "themes": [], "timestamp": 1700412035.0, "notable_quotes": []},
    ]
    driver = MockDriver(mock_data)
    engine = RetrievalEngine(driver)
    results = engine.retrieve(entities=["beach"], max_results=5)
    assert len(results) == 2
    assert results[0].memory_id == "mem-1"


def test_structured_query_by_emotion():
    mock_data = [
        {"memory_id": "mem-3", "text": "We laughed so hard",
         "summary": "Laughing fit", "relevance_score": 0.9,
         "emotion_tags": ["chaotic_giggles"], "entities": [],
         "themes": [], "timestamp": 1700412036.0, "notable_quotes": []},
    ]
    driver = MockDriver(mock_data)
    engine = RetrievalEngine(driver)
    results = engine.query(emotion_tags=["chaotic_giggles"], max_results=10)
    assert len(results) == 1
    assert "chaotic_giggles" in results[0].emotion_tags


def test_text_search():
    mock_data = [
        {"memory_id": "mem-10", "text": "Pickle the cat is awesome",
         "summary": "About Pickle", "relevance_score": 1.0,
         "emotion_tags": [], "entities": ["Pickle"],
         "themes": [], "timestamp": 1700412037.0, "notable_quotes": []},
        {"memory_id": "mem-11", "text": "My cat Pickle loves naps",
         "summary": "Pickle napping", "relevance_score": 0.9,
         "emotion_tags": [], "entities": ["Pickle"],
         "themes": [], "timestamp": 1700412038.0, "notable_quotes": []},
    ]
    driver = MockDriver(mock_data)
    engine = RetrievalEngine(driver)
    results = engine.search_text(term="Pickle", max_results=10)
    assert len(results) == 2
    assert all("Pickle" in r.text or "Pickle" in r.summary for r in results)


def test_serendipity_parameter():
    driver = MockDriver([])
    engine = RetrievalEngine(driver, serendipity=0.5)
    assert engine.serendipity == 0.5


def test_reminds_me_of():
    mock_data = [
        {"memory_id": "mem-20", "text": "That time with the crabs",
         "summary": "Crab memory", "relevance_score": 0.7,
         "emotion_tags": ["nostalgic"], "entities": ["crab"],
         "themes": [], "timestamp": 1700412039.0, "notable_quotes": []},
    ]
    driver = MockDriver(mock_data)
    engine = RetrievalEngine(driver)
    results = engine.reminds_me_of(current_topic="beach", max_results=3)
    # Even with mock, should not crash
    assert isinstance(results, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

Create `vibe_memory/retrieval/engine.py`:

```python
"""Retrieval Engine — context-aware, vibe-first memory retrieval."""

import logging
from datetime import datetime, timezone
from typing import Optional

from vibe_memory.models import Memory

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Retrieves memories from the Neo4j graph."""

    def __init__(self, driver, serendipity: float = 0.15):
        """
        Args:
            driver: Neo4j driver instance
            serendipity: 0.0 (strict relevance) to 1.0 (maximum chaos)
        """
        self.driver = driver
        self.serendipity = max(0.0, min(1.0, serendipity))

    def retrieve(
        self,
        context: Optional[str] = None,
        entities: Optional[list[str]] = None,
        emotion_filter: Optional[list[str]] = None,
        max_results: int = 5,
        recency_bias: str = "balanced",
        serendipity: Optional[float] = None,
    ) -> list[Memory]:
        """Retrieve memories based on context, entities, and emotions.

        Args:
            context: Natural language context (extracted to entities/emotions by LLM)
            entities: Explicit entity list (e.g., ["beach", "crab"])
            emotion_filter: Filter by emotion tags
            max_results: Maximum memories to return
            recency_bias: "recent", "balanced", or "historical"
            serendipity: Override instance serendipity for this call

        Returns:
            Ranked list of Memory objects
        """
        serendipity = serendipity if serendipity is not None else self.serendipity

        # Build query clauses
        where_clauses = []
        params = {}

        if entities:
            where_clauses.append("ANY(e IN $entities WHERE e IN m.entities)")
            params["entities"] = entities

        if emotion_filter:
            where_clauses.append("ANY(em IN $emotions WHERE em IN m.emotion_tags)")
            params["emotions"] = emotion_filter

        # Recency filtering
        now = datetime.now(timezone.utc).timestamp()
        if recency_bias == "recent":
            where_clauses.append("m.timestamp > $recent_cutoff")
            params["recent_cutoff"] = now - (30 * 86400)
        elif recency_bias == "historical":
            where_clauses.append("m.timestamp < $old_cutoff")
            params["old_cutoff"] = now - (90 * 86400)

        where_str = " AND ".join(where_clauses) if where_clauses else "TRUE"
        params["now"] = now
        params["max_results"] = max_results

        query = f"""
            MATCH (m:Memory)
            WHERE {where_str}
            ORDER BY m.relevance_score DESC, m.timestamp DESC
            LIMIT $max_results
            RETURN m {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                        .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """

        with self.driver.session() as session:
            result = session.run(query, params)
            memories = [self._record_to_memory(r["data"]) for r in result]

        # Add serendipity results if enabled
        if serendipity > 0 and entities:
            serendipity_count = max(1, int(max_results * serendipity))
            vibe_memories = self._vibe_walk(entities, serendipity_count)
            existing_ids = {m.memory_id for m in memories}
            for vm in vibe_memories:
                if vm.memory_id not in existing_ids and len(memories) < max_results:
                    memories.append(vm)
                    existing_ids.add(vm.memory_id)

        return memories

    def query(
        self,
        emotion_tags: Optional[list[str]] = None,
        entities: Optional[list[str]] = None,
        themes: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        max_results: int = 10,
    ) -> list[Memory]:
        """Structured query for precise control.

        Args:
            emotion_tags: Filter by emotion tags
            entities: Filter by entities
            themes: Filter by themes
            since: Only memories after this time
            until: Only memories before this time
            max_results: Maximum results
        """
        where_clauses = []
        params = {"max_results": max_results}

        if emotion_tags:
            where_clauses.append("ANY(em IN $emotion_tags WHERE em IN m.emotion_tags)")
            params["emotion_tags"] = emotion_tags
        if entities:
            where_clauses.append("ANY(e IN $entities WHERE e IN m.entities)")
            params["entities"] = entities
        if themes:
            where_clauses.append("ANY(t IN $themes WHERE t IN m.themes)")
            params["themes"] = themes
        if since:
            where_clauses.append("m.timestamp >= $since")
            params["since"] = since.timestamp()
        if until:
            where_clauses.append("m.timestamp <= $until")
            params["until"] = until.timestamp()

        where_str = " AND ".join(where_clauses) if where_clauses else "TRUE"

        query = f"""
            MATCH (m:Memory)
            WHERE {where_str}
            ORDER BY m.relevance_score DESC, m.timestamp DESC
            LIMIT $max_results
            RETURN m {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                        .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """

        with self.driver.session() as session:
            result = session.run(query, params)
            return [self._record_to_memory(r["data"]) for r in result]

    def search_text(
        self,
        term: str,
        max_results: int = 200,
        sort_by: str = "relevance",
    ) -> list[Memory]:
        """Full-text search across memory content.

        Args:
            term: Search term (literal string match)
            max_results: Maximum results
            sort_by: "relevance", "recency", or "frequency"
        """
        if sort_by == "recency":
            order_expr = "m.timestamp DESC"
        elif sort_by == "frequency":
            order_expr = "size(split(m.text, $term)) DESC, m.relevance_score DESC"
        else:
            order_expr = "m.relevance_score DESC, m.timestamp DESC"

        query = f"""
            MATCH (m:Memory)
            WHERE toLower(m.text) CONTAINS toLower($term)
               OR toLower(m.summary) CONTAINS toLower($term)
            ORDER BY {order_expr}
            LIMIT $max_results
            RETURN m {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                        .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """

        with self.driver.session() as session:
            result = session.run(query, {"term": term, "max_results": max_results})
            return [self._record_to_memory(r["data"]) for r in result]

    def _vibe_walk(self, entities: list[str], max_results: int) -> list[Memory]:
        """2-hop vibe walk through entity edges for serendipity."""
        query = """
            MATCH (m1:Memory)<-[:MENTIONS]-(e:Entity)-[:MENTIONS]->(m2:Memory)
            WHERE ANY(ent IN $entities WHERE ent IN m1.entities)
              AND m1.memory_id <> m2.memory_id
            WITH m2, count(*) AS connections
            ORDER BY connections DESC, m2.relevance_score DESC
            LIMIT $max_results
            RETURN m2 {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                          .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, {"entities": entities, "max_results": max_results})
                return [self._record_to_memory(r["data"]) for r in result]
        except Exception as e:
            logger.warning(f"Vibe walk failed: {e}")
            return []

    def reminds_me_of(
        self,
        current_topic: str,
        max_hops: int = 3,
        serendipity: float = 0.4,
        max_results: int = 3,
    ) -> list[Memory]:
        """Find memories that remind you of a topic, with deep traversal."""
        # Build variable-length path pattern
        hop_pattern = "-[:MENTIONS|HAS_EMOTION*1.." + str(max_hops) + "]-"
        query = f"""
            MATCH (m1:Memory)
            WHERE toLower(m1.text) CONTAINS toLower($topic)
                   OR toLower(m1.summary) CONTAINS toLower($topic)
                   OR ANY(e IN m1.entities WHERE toLower(e) CONTAINS toLower($topic))
            MATCH path = m1 {hop_pattern} m2:Memory
            WHERE m1.memory_id <> m2.memory_id
            WITH DISTINCT m2
            ORDER BY m2.relevance_score DESC
            LIMIT $max_results
            RETURN m2 {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                          .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, {"topic": current_topic, "max_results": max_results})
                return [self._record_to_memory(r["data"]) for r in result]
        except Exception as e:
            logger.warning(f"reminds_me_of failed: {e}")
            return []

    def boost_memory(self, memory_id: str, delta: float = 0.1):
        """Boost a memory's relevance score (revival)."""
        with self.driver.session() as session:
            session.run("""
                MATCH (m:Memory {memory_id: $mem_id})
                SET m.relevance_score = MIN(COALESCE(m.relevance_score, 1.0) + $delta, 2.0)
            """, {"mem_id": memory_id, "delta": delta})

    def decay_all(self, delta: float = 0.01):
        """Decay all memory relevance scores slightly."""
        with self.driver.session() as session:
            session.run("""
                MATCH (m:Memory)
                SET m.relevance_score = MAX(COALESCE(m.relevance_score, 1.0) - $delta, 0.0)
            """, {"delta": delta})

    def _record_to_memory(self, data: dict) -> Memory:
        """Convert a Neo4j record dict to a Memory object."""
        ts = None
        if data.get("timestamp"):
            try:
                ts = datetime.fromtimestamp(data["timestamp"], tz=timezone.utc)
            except (ValueError, TypeError, OSError):
                ts = None
        return Memory(
            memory_id=data.get("memory_id", ""),
            text=data.get("text", ""),
            summary=data.get("summary", ""),
            entities=data.get("entities", []),
            emotion_tags=data.get("emotion_tags", []),
            themes=data.get("themes", []),
            timestamp=ts,
            relevance_score=data.get("relevance_score", 1.0),
            notable_quotes=data.get("notable_quotes", []),
        )


class MemoryStore:
    """High-level API for the Vibe Memory system.

    This is the main entry point for waifu/husbando bots.

    Usage:
        store = MemoryStore(neo4j_url="bolt://localhost:7687")
        memories = store.retrieve(context="we're at the beach", serendipity=0.15)
    """

    def __init__(self, neo4j_url: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", neo4j_password: str = "password",
                 serendipity: float = 0.15):
        from neo4x import GraphDatabase
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        self.engine = RetrievalEngine(self.driver, serendipity=serendipity)

    def retrieve(self, **kwargs) -> list[Memory]:
        """Context-based retrieval."""
        return self.engine.retrieve(**kwargs)

    def query(self, **kwargs) -> list[Memory]:
        """Structured query."""
        return self.engine.query(**kwargs)

    def search_text(self, **kwargs) -> list[Memory]:
        """Full-text search."""
        return self.engine.search_text(**kwargs)

    def reminds_me_of(self, **kwargs) -> list[Memory]:
        """Deep traversal for unexpected connections."""
        return self.engine.reminds_me_of(**kwargs)

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

Modify `vibe_memory/retrieval/__init__.py`:

```python
from vibe_memory.retrieval.engine import RetrievalEngine, MemoryStore

__all__ = ["RetrievalEngine", "MemoryStore"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: PASS — all 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add vibe_memory/retrieval/ tests/test_retrieval.py
git commit -m "feat: retrieval engine — context, structured, and text search

- RetrievalEngine: retrieve(), query(), search_text(), reminds_me_of()
- Vibe walk: 2-hop traversal through entity edges for serendipity
- Serendipity knob (0.0-1.0) controls chaos vs relevance ratio
- MemoryStore: high-level public API (the main entry point)
- boost_memory() and decay_all() for relevance score management
- Recency bias: recent, balanced, historical"
```

---

### Task 6: Integration — Full Pipeline + CLI

**Files:**
- Create: `vibe_memory/pipeline.py`
- Modify: `vibe_memory/__init__.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Ingestor, Summarizer, GraphBuilder from earlier tasks
- Produces: `run_ingestion()` function that orchestrates the full pipeline
- Produces: CLI entry point via `python -m vibe_memory`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import pytest
from vibe_memory.pipeline import IngestionConfig, run_ingestion


def test_ingestion_config():
    config = IngestionConfig(
        db_path="test.db",
        neo4j_url="bolt://localhost:7687",
        model_path="/path/to/model.gguf",
        chunk_size=10,
    )
    assert config.chunk_size == 10
    assert config.db_path == "test.db"


def test_run_ingestion_with_mock():
    """Test the pipeline orchestrator with mock components."""
    config = IngestionConfig(
        db_path=":memory:",  # in-memory DB
        neo4j_url="bolt://localhost:7687",
        model_path="/dummy.gguf",
        chunk_size=5,
    )
    # Should not crash even with empty DB and no model
    result = run_ingestion(config, dry_run=True)
    assert result["conversations_processed"] >= 0
    assert result["errors"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

Create `vibe_memory/pipeline.py`:

```python
"""Pipeline — orchestrates the full ingestion pipeline."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from vibe_memory.ingestor import Ingestor
from vibe_memory.summarizer.base import Summarizer
from vibe_memory.summarizer.llama_cpp import LlamaCppSummarizer
from vibe_memory.graph.builder import GraphBuilder
from vibe_memory.models import Memory, Conversation, Chapter, Detail

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    db_path: str
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    model_path: str = ""
    chunk_size: int = 10
    summarizer_backend: str = "llama_cpp"  # llama_cpp, ollama, openai
    dry_run: bool = False
    chapter_grouping: str = "monthly"  # monthly, yearly, manual


async def run_ingestion(config: IngestionConfig, summarizer: Optional[Summarizer] = None) -> dict:
    """Run the full ingestion pipeline.

    Pipeline: Ingestor → Summarizer → GraphBuilder → Neo4j

    Args:
        config: Ingestion configuration
        summarizer: Optional pre-configured summarizer (for testing)

    Returns:
        Stats dict with counts and error information
    """
    stats = {
        "conversations_processed": 0,
        "memories_created": 0,
        "chapters_created": 0,
        "errors": 0,
    }

    # Initialize components
    if config.dry_run:
        logger.info("Dry run mode — no Neo4j or LLM calls")
        return stats

    # Open connections
    from neo4x import GraphDatabase
    driver = GraphDatabase.driver(config.neo4j_url, auth=(config.neo4j_user, config.neo4j_password))
    builder = GraphBuilder(driver)

    try:
        # Create schema
        builder.create_schema()

        # Initialize summarizer
        if summarizer is None:
            summarizer = LlamaCppSummarizer(model_path=config.model_path)

        # Open ingestor
        with Ingestor(config.db_path, chunk_size=config.chunk_size) as ingestor:
            conversations = ingestor.load_conversations()
            logger.info(f"Loaded {len(conversations)} conversations")

            # Group conversations into chapters
            chapters = _group_into_chapters(conversations, config.chapter_grouping)

            for chapter in chapters:
                # Summarize chapter
                try:
                    conv_summaries = []
                    for conv in chapter.conversations:
                        # Summarize conversation
                        chunks = ingestor.chunk_conversation(conv)
                        chunks_text = [ingestor._messages_to_text(c) for c in chunks]

                        try:
                            conv_summary = await summarizer.summarize_conversation(chunks_text)
                            conv_summaries.append(conv_summary)
                        except Exception as e:
                            logger.warning(f"Failed to summarize conversation {conv.conversation_id}: {e}")
                            stats["errors"] += 1
                            conv_summaries.append(
                                type(conv_summary)(title=conv.title, summary=conv.title)
                            )

                    # Build chapter summary
                    try:
                        chapter_summary = await summarizer.summarize_chapter(conv_summaries)
                    except Exception as e:
                        logger.warning(f"Failed to summarize chapter: {e}")
                        stats["errors"] += 1
                        chapter_summary = type(chapter_summary)(title=chapter.title)

                    # Add chapter to graph
                    chapter_node = Chapter(
                        chapter_id=chapter.chapter_id,
                        title=chapter_summary.title or chapter.title,
                        start_time=chapter.start_time,
                        end_time=chapter.end_time,
                        summary=chapter_summary.summary,
                        themes=chapter_summary.recurring_themes,
                        emotion_tags=[],
                    )
                    builder.add_chapter(chapter_node)
                    stats["chapters_created"] += 1

                    # Process each conversation
                    for conv in chapter.conversations:
                        chunks = ingestor.chunk_conversation(conv)
                        for chunk_idx, chunk in enumerate(chunks):
                            chunk_text = ingestor._messages_to_text(chunk)

                            # Extract memory
                            try:
                                extraction = await summarizer.extract_memory(chunk_text)
                            except Exception as e:
                                logger.warning(f"Failed to extract memory: {e}")
                                stats["errors"] += 1
                                extraction = type(extraction)(
                                    summary=chunk_text[:200],
                                )

                            # Create memory node
                            memory = Memory(
                                memory_id=str(uuid.uuid4()),
                                text=chunk_text,
                                summary=extraction.summary,
                                entities=extraction.entities,
                                emotion_tags=extraction.emotion_tags,
                                themes=extraction.themes,
                                timestamp=datetime.now(timezone.utc),
                                relevance_score=1.0,
                                notable_quotes=extraction.notable_quotes,
                            )
                            builder.add_memory(memory, conv.conversation_id)
                            stats["memories_created"] += 1

                        stats["conversations_processed"] += 1

        logger.info(f"Ingestion complete: {stats}")
    finally:
        builder.close()

    return stats


def _group_into_chapters(conversations: list[Conversation], grouping: str):
    """Group conversations into chapters by time period."""
    from dataclasses import dataclass

    @dataclass
    class ChapterGroup:
        chapter_id: str
        title: str
        start_time: float
        end_time: float
        conversations: list[Conversation] = field(default_factory=list)

    if not conversations:
        return []

    chapters = []
    current_chapter = None

    for conv in conversations:
        if grouping == "monthly":
            # Group by year-month
            dt = datetime.fromtimestamp(conv.create_time, tz=timezone.utc)
            period_key = dt.strftime("%Y-%m")
        elif grouping == "yearly":
            dt = datetime.fromtimestamp(conv.create_time, tz=timezone.utc)
            period_key = dt.strftime("%Y")
        else:
            period_key = "all"

        if current_chapter is None or current_chapter.period_key != period_key:
            if current_chapter:
                chapters.append(current_chapter)
            dt = datetime.fromtimestamp(conv.create_time, tz=timezone.utc)
            current_chapter = ChapterGroup(
                chapter_id=str(uuid.uuid4()),
                title=period_key,
                start_time=conv.create_time,
                end_time=conv.create_time,
                period_key=period_key,
            )

        current_chapter.conversations.append(conv)
        current_chapter.end_time = max(current_chapter.end_time, conv.create_time)

    if current_chapter:
        chapters.append(current_chapter)

    return chapters
```

Modify `vibe_memory/__init__.py`:

```python
"""Vibe Memory — memory graph for AI companion bots."""

from vibe_memory.models import (
    Memory, Conversation, Chapter, Detail,
    MemoryExtraction, ConversationSummary, ChapterSummary,
)
from vibe_memory.retrieval.engine import MemoryStore
from vibe_memory.pipeline import IngestionConfig, run_ingestion

__all__ = [
    "Memory", "Conversation", "Chapter", "Detail",
    "MemoryExtraction", "ConversationSummary", "ChapterSummary",
    "MemoryStore", "IngestionConfig", "run_ingestion",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS — both tests pass

- [ ] **Step 5: Commit**

```bash
git add vibe_memory/pipeline.py vibe_memory/__init__.py tests/test_pipeline.py
git commit -m "feat: integration pipeline — full ingestion orchestrator

- run_ingestion(): Ingestor → Summarizer → GraphBuilder → Neo4j
- IngestionConfig with all pipeline options
- Chapter grouping by monthly/yearly
- Non-fatal error handling: skip and continue
- Dry run mode for testing
- MemoryStore exported as main public API"
```

---

### Task 7: Self-Review and Final Polish

- [ ] **Step 1: Spec coverage check**

Verify every spec requirement has a task:
- ✅ Project structure + data models (Task 1)
- ✅ Ingestor: chats.db → chunks (Task 2)
- ✅ Summarizer: base + llama-cpp (Task 3)
- ✅ Graph builder: Neo4j schema + nodes (Task 4)
- ✅ Retrieval: context + structured + text search (Task 5)
- ✅ Vibe walk + serendipity (Task 5)
- ✅ Integration pipeline (Task 6)
- ✅ Error handling: non-fatal, resumable (Tasks 3, 6)
- ✅ `chaos/` and `dream/` folders (Task 1)
- ✅ `[:SARAH_APPROVED]` edge type (Task 4)

- [ ] **Step 2: Placeholder scan**

No TBDs, no TODOs, no "implement later". All code blocks contain actual implementations.

- [ ] **Step 3: Type consistency**

All method signatures match across tasks:
- `Summarizer.extract_memory(text: str) -> MemoryExtraction` ✅
- `GraphBuilder.add_memory(memory: Memory, conversation_id: str)` ✅
- `RetrievalEngine.retrieve(...) -> list[Memory]` ✅
- `MemoryStore` delegates to `RetrievalEngine` ✅

- [ ] **Step 4: Run all tests**

```bash
cd /home/toxi/c/archive
pip install -e ".[dev]" neo4x
python -m pytest tests/ -v
```

Expected: All tests pass across all modules.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: vibe memory system complete

- Full pipeline: ingest → summarize → graph → retrieve
- Neo4j graph with hierarchical schema
- Local LLM summarization (llama-cpp, pluggable backends)
- Context-based + structured + text search retrieval
- Serendipity knob (0.0-1.0) with vibe walk
- Emotion tags, entity tracking, relevance scoring
- chaos/ and dream/ folders for experimental features"
```

---

## Execution Handoff

**Plan complete. Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session with checkpoints

**Which approach?**
</parameter>user ok, so it's a go! ?? let's invoke the writing-plans skill to create the implementation plan. <tool_call>read <tool_call>bash  <tool_call>edit  Successfully replaced 1 block in /home/toxi/c/archive/docs/superpowers/plans/2026-06-22-vibe-memory.md.  Now let me do a self-review of the plan and check for any issues.  <tool_call>bash  tail -10 /home/toxi/c/archive/docs/superpowers/plans/2026-06-22-vibe-memory.md  class MemoryStore:  