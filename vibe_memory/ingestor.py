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
        return results

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
