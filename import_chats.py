#!/usr/bin/env python3
"""
Import extracted ChatGPT chat JSON files into a SQLite database.

Usage:
    python3 import_chats.py /mnt/c/archive/extracted_chats/ output.db

Schema:
    conversations  - one row per chat file (title, timestamps, model, etc.)
    messages       - one row per message in conversation order
"""

import json
import glob
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone


def walk_messages(mapping):
    """Walk the message tree in order using parent->children links (iterative)."""
    # Find root (node with no parent or parent with no message)
    roots = []
    for nid, node in mapping.items():
        msg = node.get('message')
        if msg and msg['author']['role'] == 'system':
            roots.append(nid)
        elif msg and msg['author']['role'] == 'user' and not node.get('parent'):
            roots.append(nid)

    if not roots:
        for nid, node in mapping.items():
            if node.get('parent') and node['parent'] not in mapping:
                roots.append(nid)
                break
    if not roots:
        roots = list(mapping.keys())[:1]

    # Iterative DFS to avoid recursion limits on long conversations
    for root in roots:
        stack = [root]
        while stack:
            node_id = stack.pop()
            node = mapping.get(node_id)
            if not node:
                continue
            msg = node.get('message')
            if msg:
                yield msg
            # Push children in reverse so first child is processed first
            for child_id in reversed(node.get('children', [])):
                stack.append(child_id)


def extract_text(content):
    """Extract text from a content block."""
    ct = content.get('content_type', 'text')
    if ct == 'text':
        return ' '.join(content.get('parts', []))
    elif ct == 'multimodal_text':
        parts = content.get('parts', [])
        texts = []
        for p in parts:
            if isinstance(p, str):
                texts.append(p)
            elif isinstance(p, dict):
                if p.get('type') == 'text':
                    texts.append(p.get('text', ''))
                elif p.get('type') == 'image_url':
                    url = p.get('image_url', {})
                    if isinstance(url, dict):
                        texts.append(f"[image: {url.get('url', '')}]")
                    else:
                        texts.append(f"[image: {url}]")
        return ' '.join(texts)
    elif ct == 'code':
        return content.get('text', '')
    elif ct == 'execution_output':
        return content.get('text', '')
    elif ct == 'system_error':
        return content.get('text', '')
    elif ct == 'tether_browsing_display':
        return content.get('text', '')
    elif ct == 'tether_quote':
        return content.get('text', '')
    return json.dumps(content)


def extract_model_slug(messages):
    """Try to find the model slug from assistant messages."""
    for msg in messages:
        slug = msg.get('metadata', {}).get('model_slug')
        if slug:
            return slug
    return None


def create_schema(conn):
    conn.executescript("""
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
        );

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
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, message_index);
        CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
        CREATE INDEX IF NOT EXISTS idx_conversations_model ON conversations(model_slug);
        CREATE INDEX IF NOT EXISTS idx_conversations_create_time ON conversations(create_time);
    """)


def import_chat(conn, filepath):
    """Import a single chat JSON file into the database."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conversation_id = data.get('conversation_id', data.get('id', ''))
    title = data.get('title', '')
    create_time = data.get('create_time')
    update_time = data.get('update_time')
    is_archived = 1 if data.get('is_archived') else 0
    current_node = data.get('current_node')
    plugin_ids = json.dumps(data.get('plugin_ids')) if data.get('plugin_ids') else None
    gizmo_id = data.get('gizmo_id')
    conversation_template_id = data.get('conversation_template_id')

    create_time_iso = datetime.fromtimestamp(create_time, tz=timezone.utc).isoformat() if create_time else None
    update_time_iso = datetime.fromtimestamp(update_time, tz=timezone.utc).isoformat() if update_time else None

    # Walk messages in order
    messages = list(walk_messages(data.get('mapping', {})))

    # Filter out hidden system messages
    visible_messages = [
        m for m in messages
        if not (m.get('metadata', {}).get('is_visually_hidden_from_conversation'))
    ]

    model_slug = extract_model_slug(visible_messages)

    conn.execute("""
        INSERT OR REPLACE INTO conversations
        (conversation_id, title, create_time, create_time_iso, update_time, update_time_iso,
         model_slug, message_count, is_archived, current_node, plugin_ids, gizmo_id,
         conversation_template_id, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        conversation_id, title, create_time, create_time_iso, update_time, update_time_iso,
        model_slug, len(visible_messages), is_archived, current_node,
        plugin_ids, gizmo_id, conversation_template_id, os.path.basename(filepath)
    ))

    for idx, msg in enumerate(visible_messages):
        content = msg.get('content', {})
        text = extract_text(content)
        content_type = content.get('content_type', 'text')
        role = msg['author']['role']
        msg_create_time = msg.get('create_time')
        msg_create_time_iso = datetime.fromtimestamp(msg_create_time, tz=timezone.utc).isoformat() if msg_create_time else None
        metadata = msg.get('metadata', {})
        msg_model = metadata.get('model_slug')
        code_language = content.get('language') if content_type == 'code' else None

        # Find parent message id
        node_id = msg['id']
        parent_id = None
        node = data.get('mapping', {}).get(node_id)
        if node:
            parent_id = node.get('parent')

        conn.execute("""
            INSERT OR REPLACE INTO messages
            (message_id, conversation_id, message_index, role, content_type, content,
             create_time, create_time_iso, update_time, status, weight, model_slug,
             code_language, parent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            msg['id'], conversation_id, idx, role, content_type, text,
            msg_create_time, msg_create_time_iso, msg.get('update_time'),
            msg.get('status'), msg.get('weight'), msg_model,
            code_language, parent_id
        ))


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_directory> <output.db>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_db = sys.argv[2]

    # Remove existing DB to start fresh
    if os.path.exists(output_db):
        os.remove(output_db)

    conn = sqlite3.connect(output_db)
    create_schema(conn)

    files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    print(f"Found {len(files)} chat files to import...")

    start = time.time()
    errors = 0
    for i, filepath in enumerate(files):
        try:
            import_chat(conn, filepath)
            if (i + 1) % 20 == 0:
                print(f"  Imported {i + 1}/{len(files)}...")
        except Exception as e:
            errors += 1
            print(f"  Error importing {os.path.basename(filepath)}: {e}")

    conn.commit()
    elapsed = time.time() - start

    # Print summary
    cursor = conn.execute("SELECT COUNT(*) FROM conversations")
    conv_count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM messages")
    msg_count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT model_slug, COUNT(*) FROM conversations WHERE model_slug IS NOT NULL GROUP BY model_slug ORDER BY COUNT(*) DESC")
    print(f"\nImport complete in {elapsed:.1f}s")
    print(f"  Conversations: {conv_count}")
    print(f"  Messages: {msg_count}")
    print(f"  Errors: {errors}")
    print(f"\nModel distribution:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} conversations")

    cursor = conn.execute("SELECT role, COUNT(*) FROM messages GROUP BY role")
    print(f"\nMessage roles:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cursor = conn.execute("SELECT content_type, COUNT(*) FROM messages GROUP BY content_type")
    print(f"\nContent types:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    conn.close()
    print(f"\nDatabase saved to: {output_db}")


if __name__ == '__main__':
    main()
