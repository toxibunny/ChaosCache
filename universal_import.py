#!/usr/bin/env python3
"""Universal Chat Importer — auto-detects and imports from various chat archive formats.

Handles known formats (ChatGPT, Claude, Discord, WhatsApp, etc.) and falls back
to LLM-based schema detection for unknown formats.

Usage:
    python3 universal_import.py /path/to/chats/ output.db
    python3 universal_import.py /path/to/chats/ output.db --model /path/to/model.gguf
"""

import json
import logging
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import jmespath

logger = logging.getLogger(__name__)


# ─── Known Format Parsers ───────────────────────────────────────────────────


@dataclass
class Message:
    message_id: str
    conversation_id: str
    message_index: int
    role: str
    content_type: str
    content: str
    create_time: float
    parent_id: Optional[str] = None


@dataclass
class Conversation:
    conversation_id: str
    title: str
    create_time: float
    update_time: float
    model_slug: Optional[str]
    messages: list[Message]


class FormatDetector:
    """Detects chat archive format and returns a parser."""

    KNOWN_FORMATS = {
        "chatgpt": "ChatGPT/ChatGPT.neural json exports",
        "claude": "Claude conversation exports",
        "discord": "Discord chat exports (BetterDiscord, etc.)",
        "whatsapp": "WhatsApp chat exports",
        "telegram": "Telegram chat exports",
        "generic_json": "Generic JSON with messages array",
    }

    def detect(self, file_path: Path) -> str:
        """Detect the format of a chat file."""
        try:
            with open(file_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "unknown"

        if isinstance(data, dict):
            # ChatGPT format (has mapping or messages)
            if "mapping" in data and isinstance(data["mapping"], dict):
                return "chatgpt"
            if "messages" in data:
                msgs = data["messages"]
                if isinstance(msgs, list) and msgs and "role" in msgs[0]:
                    return "chatgpt"
                if isinstance(msgs, dict):
                    return "chatgpt"
            # Claude format
            if "conversation_persistent_id" in data or "message_groups" in data:
                return "claude"
            # Discord format
            if "channel" in data:
                return "discord"

        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                if "role" in data[0] and "content" in data[0]:
                    return "generic_json"
                if "body" in data[0] and "from" in data[0]:
                    return "whatsapp"
                if "date" in data[0] and "from" in data[0]:
                    return "telegram"

        return "unknown"


# ─── Parsers ─────────────────────────────────────────────────────────────────


def parse_chatgpt(file_path: Path) -> list[Conversation]:
    """Parse ChatGPT/ChatGPT.neural JSON export."""
    with open(file_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        # Array of conversations
        conversations = []
        for conv_data in data:
            conv = _parse_chatgpt_conversation(conv_data, file_path.stem)
            if conv:
                conversations.append(conv)
        return conversations
    elif isinstance(data, dict):
        # Single conversation
        conv = _parse_chatgpt_conversation(data, file_path.stem)
        return [conv] if conv else []
    return []


def _parse_chatgpt_conversation(data: dict, default_id: str) -> Optional[Conversation]:
    """Parse a single ChatGPT conversation."""
    conv_id = data.get("id", data.get("conversation_id", default_id))
    title = data.get("title", data.get("gizmo_title", "Untitled"))
    create_time = data.get("create_time", data.get("created_at", 0))
    update_time = data.get("update_time", data.get("updated_at", create_time))
    model_slug = data.get("model_slug", data.get("model", None))

    # Handle messages array or mapping dict
    messages_raw = data.get("messages", [])
    if not messages_raw and isinstance(data.get("mapping"), dict):
        # ChatGPT.neural format with mapping dict
        messages_raw = list(data["mapping"].values())
    elif not messages_raw and isinstance(data.get("nodes"), dict):
        # Alternative nodes format
        messages_raw = list(data["nodes"].values())

    messages = []
    for idx, msg_data in enumerate(messages_raw):
        if not isinstance(msg_data, dict):
            continue

        role = msg_data.get("role", msg_data.get("sender", "user"))
        content = _extract_content(msg_data)
        msg_id = msg_data.get("id", str(uuid.uuid4()))
        msg_time = msg_data.get("create_time", msg_data.get("timestamp", create_time))
        parent_id = msg_data.get("parent", msg_data.get("parent_id"))

        messages.append(Message(
            message_id=msg_id,
            conversation_id=conv_id,
            message_index=idx,
            role=role,
            content_type=msg_data.get("content_type", "text"),
            content=content,
            create_time=msg_time,
            parent_id=parent_id,
        ))

    return Conversation(
        conversation_id=conv_id,
        title=title,
        create_time=create_time,
        update_time=update_time,
        model_slug=model_slug,
        messages=messages,
    )


def _extract_content(msg_data: dict) -> str:
    """Extract text content from a message."""
    content = msg_data.get("content", "")
    if isinstance(content, dict):
        # ChatGPT content structure
        parts = content.get("parts", [])
        if parts:
            return "\n".join(parts)
        return json.dumps(content)
    if isinstance(content, list):
        return "\n".join(str(p) for p in content)
    return str(content) if content else ""


def parse_generic_json(file_path: Path) -> list[Conversation]:
    """Parse generic JSON with messages array."""
    with open(file_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        # Array of messages (single conversation)
        messages = []
        for idx, msg_data in enumerate(data):
            content = msg_data.get("content", msg_data.get("text", msg_data.get("message", "")))
            role = msg_data.get("role", msg_data.get("type", "user"))
            msg_time = msg_data.get("timestamp", msg_data.get("time", msg_data.get("date", 0)))

            messages.append(Message(
                message_id=msg_data.get("id", str(uuid.uuid4())),
                conversation_id=file_path.stem,
                message_index=idx,
                role=role,
                content_type="text",
                content=str(content),
                create_time=_to_timestamp(msg_time),
            ))

        return [Conversation(
            conversation_id=file_path.stem,
            title=file_path.stem,
            create_time=messages[0].create_time if messages else 0,
            update_time=messages[-1].create_time if messages else 0,
            model_slug=None,
            messages=messages,
        )]
    return []


def parse_with_llm(file_path: Path, model_path: str) -> list[Conversation]:
    """Parse an unknown format using LLM-based schema detection."""
    with open(file_path) as f:
        data = json.load(f)

    # Extract structure
    structure = _extract_structure(data)
    mapping = _detect_schema_with_llm(structure, model_path)

    if not mapping:
        logger.warning(f"LLM failed to detect schema for {file_path}")
        return []

    # Apply mapping
    conversations = _apply_mapping(data, mapping, file_path.stem)
    return conversations


def _extract_structure(data: Any, max_depth: int = 5) -> dict:
    """Extract the structure of a JSON object."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(value, list) and value:
                result[key] = f"array[{len(value)}] of {_type_name(value[0])}"
            elif isinstance(value, dict):
                result[key] = _extract_structure(value, max_depth - 1) if max_depth > 0 else "object"
            else:
                result[key] = _type_name(value)
        return result
    elif isinstance(data, list) and data:
        return f"array[{len(data)}] of {_extract_structure(data[0], max_depth - 1) if max_depth > 0 else _type_name(data[0])}"
    return _type_name(data)


def _type_name(value: Any) -> str:
    """Get a human-readable type name."""
    if isinstance(value, str):
        return f'string("{value[:50]}...")' if len(value) > 50 else f'string("{value}")'
    if isinstance(value, (int, float)):
        return f"number({value})"
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    return type(value).__name__


def _detect_schema_with_llm(structure: dict, model_path: str) -> dict:
    """Use LLM to detect the schema mapping."""
    from vibe_memory.summarizer.llama_cpp import LlamaCppSummarizer

    summarizer = LlamaCppSummarizer(model_path=model_path)

    prompt = f"""You are a JSON schema analyzer. Given this chat archive structure, output a JSON mapping to extract conversations and messages.

Structure:
{json.dumps(structure, indent=2)}

Output a JSON object with this exact structure:
{{
    "conversation_id": "jmespath to conversation id",
    "title": "jmespath to title",
    "create_time": "jmespath to timestamp",
    "messages_path": "jmespath to messages array",
    "message_id": "jmespath to message id (within array)",
    "role": "jmespath to role",
    "content": "jmespath to content",
    "timestamp": "jmespath to message timestamp"
}}

For nested arrays, use @ to refer to the current message item. Example: if messages is at "conversations.messages", then message fields use "@.id", "@.role", "@.content"."""

    try:
        result = summarizer._call_llm(prompt)
        # Extract JSON from response
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(result[start:end])
    except Exception as e:
        logger.warning(f"LLM schema detection failed: {e}")

    return {}


def _apply_mapping(data: Any, mapping: dict, default_id: str) -> list[Conversation]:
    """Apply a JMESPath mapping to extract conversations."""
    # Extract conversations (might be a single object or array)
    conv_id_path = mapping.get("conversation_id", "")
    title_path = mapping.get("title", "")
    create_time_path = mapping.get("create_time", "")
    messages_path = mapping.get("messages_path", "")

    msg_id_path = mapping.get("message_id", "@.id")
    role_path = mapping.get("role", "@.role")
    content_path = mapping.get("content", "@.content")
    timestamp_path = mapping.get("timestamp", "@.timestamp")

    # Handle both single conversation and array of conversations
    if isinstance(data, dict):
        conversations_data = [data]
    elif isinstance(data, list):
        conversations_data = data
    else:
        return []

    conversations = []
    for conv_data in conversations_data:
        # Extract conversation-level fields
        conv_id = jmespath.search(conv_id_path, conv_data) or default_id
        title = jmespath.search(title_path, conv_data) or default_id
        create_time = jmespath.search(create_time_path, conv_data) or 0

        # Extract messages
        messages_raw = jmespath.search(messages_path, conv_data) or []
        messages = []

        for idx, msg_data in enumerate(messages_raw):
            msg_id = jmespath.search(msg_id_path, msg_data) or str(uuid.uuid4())
            role = jmespath.search(role_path, msg_data) or "user"
            content = jmespath.search(content_path, msg_data) or ""
            msg_time = jmespath.search(timestamp_path, msg_data) or create_time

            messages.append(Message(
                message_id=str(msg_id),
                conversation_id=str(conv_id),
                message_index=idx,
                role=str(role),
                content_type="text",
                content=str(content) if content else "",
                create_time=_to_timestamp(msg_time),
            ))

        conversations.append(Conversation(
            conversation_id=str(conv_id),
            title=str(title),
            create_time=_to_timestamp(create_time),
            update_time=_to_timestamp(create_time),
            model_slug=None,
            messages=messages,
        ))

    return conversations


def _to_timestamp(value: Any) -> float:
    """Convert various timestamp formats to Unix timestamp."""
    if isinstance(value, (int, float)):
        # Might be milliseconds
        if value > 1e12:
            return value / 1000
        return float(value)
    if isinstance(value, str):
        # Try ISO format
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            pass
        # Try parsing as number
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


# ─── Database Writer ─────────────────────────────────────────────────────────


def create_database(db_path: str):
    """Create the SQLite database with the standard schema."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            title TEXT,
            create_time REAL,
            create_time_iso TEXT,
            update_time REAL,
            update_time_iso TEXT,
            model_slug TEXT,
            message_count INTEGER DEFAULT 0,
            is_archived INTEGER DEFAULT 0,
            current_node TEXT,
            plugin_ids TEXT,
            gizmo_id TEXT,
            conversation_template_id TEXT,
            source_file TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            content_type TEXT DEFAULT 'text',
            content TEXT,
            create_time REAL,
            create_time_iso TEXT,
            update_time REAL,
            status TEXT,
            weight REAL,
            model_slug TEXT,
            code_language TEXT,
            parent_id TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
    """)

    conn.commit()
    return conn


def write_conversations(conn: sqlite3.Connection, conversations: list[Conversation], source_file: str):
    """Write conversations and messages to the database."""
    cur = conn.cursor()
    total_messages = 0

    for conv in conversations:
        # Convert timestamps to ISO format
        create_time_iso = datetime.fromtimestamp(conv.create_time, tz=timezone.utc).isoformat() if conv.create_time else None
        update_time_iso = datetime.fromtimestamp(conv.update_time, tz=timezone.utc).isoformat() if conv.update_time else None

        cur.execute("""
            INSERT OR REPLACE INTO conversations
            (conversation_id, title, create_time, create_time_iso, update_time, update_time_iso,
             model_slug, message_count, is_archived, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, (
            conv.conversation_id,
            conv.title,
            conv.create_time,
            create_time_iso,
            conv.update_time,
            update_time_iso,
            conv.model_slug,
            len(conv.messages),
            source_file,
        ))

        for msg in conv.messages:
            msg_time_iso = datetime.fromtimestamp(msg.create_time, tz=timezone.utc).isoformat() if msg.create_time else None

            cur.execute("""
                INSERT OR REPLACE INTO messages
                (message_id, conversation_id, message_index, role, content_type, content,
                 create_time, create_time_iso, update_time, status, weight, parent_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'finished_successfully', 1.0, ?)
            """, (
                msg.message_id,
                conv.conversation_id,
                msg.message_index,
                msg.role,
                msg.content_type,
                msg.content,
                msg.create_time,
                msg_time_iso,
                msg.create_time,
                msg.parent_id,
            ))

        total_messages += len(conv.messages)

    conn.commit()
    return total_messages


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_db = sys.argv[2]
    model_path = None

    # Parse optional --model argument
    for i, arg in enumerate(sys.argv[3:]):
        if arg == "--model" and i + 1 < len(sys.argv[3:]):
            model_path = sys.argv[4 + i]
            break

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files")
    print(f"Output: {output_db}")

    # Create database
    conn = create_database(output_db)
    detector = FormatDetector()

    stats = {"conversations": 0, "messages": 0, "formats": {}, "errors": 0}

    for file_path in json_files:
        try:
            # Detect format
            fmt = detector.detect(file_path)
            stats["formats"][fmt] = stats["formats"].get(fmt, 0) + 1

            # Parse based on format
            if fmt == "chatgpt":
                conversations = parse_chatgpt(file_path)
            elif fmt == "generic_json":
                conversations = parse_generic_json(file_path)
            elif fmt == "unknown" and model_path:
                conversations = parse_with_llm(file_path, model_path)
            else:
                logger.warning(f"Skipping {file_path} (format: {fmt})")
                continue

            # Write to database
            msg_count = write_conversations(conn, conversations, str(file_path))
            stats["conversations"] += len(conversations)
            stats["messages"] += msg_count

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            stats["errors"] += 1

    conn.close()

    # Print stats
    print(f"\nImport complete:")
    print(f"  Conversations: {stats['conversations']}")
    print(f"  Messages: {stats['messages']}")
    print(f"  Formats: {', '.join(f'{k}: {v}' for k, v in stats['formats'].items())}")
    print(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
