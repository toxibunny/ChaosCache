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
            message_count INTEGER DEFAULT 0,
            is_archived INTEGER DEFAULT 0
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
    conn.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)",
                 ("conv-1", "Test Chat", 1700412034.0, "gpt-4", 4, 0))
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
