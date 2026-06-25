# Database Format

Vibe Memory ingests from a SQLite database with two tables: `conversations` and `messages`.

## Schema

### `conversations`

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | TEXT (PK) | Unique identifier (UUID) |
| `title` | TEXT | Conversation title |
| `create_time` | REAL | Unix timestamp when conversation was created |
| `create_time_iso` | TEXT | ISO 8601 timestamp |
| `update_time` | REAL | Unix timestamp of last update |
| `update_time_iso` | TEXT | ISO 8601 timestamp |
| `model_slug` | TEXT | Model used (e.g., `gpt-4`, `claude-3-opus`) |
| `message_count` | INTEGER | Number of messages |
| `is_archived` | INTEGER | 0 or 1 |
| `current_node` | TEXT | ID of the latest message node |
| `plugin_ids` | TEXT | JSON array of plugin IDs (nullable) |
| `gizmo_id` | TEXT | Gizmo/bot ID (nullable) |
| `conversation_template_id` | TEXT | Template ID (nullable) |
| `source_file` | TEXT | Original JSON file path |

### `messages`

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | TEXT (PK) | Unique identifier (UUID) |
| `conversation_id` | TEXT (FK) | References `conversations.conversation_id` |
| `message_index` | INTEGER | Position in conversation (0-based) |
| `role` | TEXT | `user`, `assistant`, or `tool` |
| `content_type` | TEXT | `text`, `code`, `multimodal_text`, etc. |
| `content` | TEXT | Message content (may be JSON for tool calls) |
| `create_time` | REAL | Unix timestamp |
| `create_time_iso` | TEXT | ISO 8601 timestamp |
| `update_time` | REAL | Unix timestamp of last update (nullable) |
| `status` | TEXT | Message status (e.g., `finished_successfully`) |
| `weight` | REAL | Message weight (default 1.0) |
| `model_slug` | TEXT | Model that generated this message (nullable) |
| `code_language` | TEXT | Language if `content_type` is `code` (nullable) |
| `parent_id` | TEXT | Parent message ID for branching conversations |

## Creating Your Own Database

### From ChatGPT JSON Exports

Use the included `import_chats.py` script:

```bash
python3 import_chats.py /path/to/extracted_chats/ chats.db
```

This handles:
- Deep message chains (tested up to 1384 nodes)
- Branching conversations
- Tool calls and multimodal content
- Model slug extraction

### From Other Sources

You can create the database manually:

```sql
CREATE TABLE conversations (
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
);

CREATE TABLE messages (
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
);
```

The ingestor only requires:
- `conversation_id`, `title`, `create_time` in `conversations`
- `conversation_id`, `message_index`, `role`, `content` in `messages`

Other columns are optional and used for metadata enrichment.

## Sample Data

```python
# conversations
('abc-123', 'Beach day', 1700412034.347, '2023-11-19T16:40:34.347+00:00', ...)

# messages
('msg-1', 'abc-123', 0, 'user', 'text', "Let's go to the beach!", 1700412034.349, ...)
('msg-2', 'abc-123', 1, 'assistant', 'text', "I'd love that! The sun, the sand...", 1700412034.350, ...)
('msg-3', 'abc-123', 2, 'tool', 'code', '{"action":"search","query":"beach weather"}', 1700412034.351, ...)
```

## Notes

- Timestamps are Unix epoch (seconds since 1970-01-01)
- Message IDs are UUIDs
- `role` determines who spoke: `user` (you), `assistant` (bot), `tool` (function call result)
- `content` may be plain text or JSON (for tool calls, image prompts, etc.)
- Branching conversations use `parent_id` to link messages in a tree structure
