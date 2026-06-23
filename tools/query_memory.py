#!/usr/bin/env python3
"""Query ChaosCache for relevant memories.

Usage:
    python3 query_memory.py NEO4J_URL [MODEL_PATH] [SERENDIPITY] [MAX_RESULTS]

Reads context from stdin, outputs JSON array of memories to stdout.
"""

import asyncio
import json
import sys

from vibe_memory.retrieval.engine import MemoryStore


async def main():
    neo4j_url = sys.argv[1] if len(sys.argv) > 1 else "bolt://localhost:7687"
    model_path = sys.argv[2] if len(sys.argv) > 2 else ""
    serendipity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    max_results = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    # Read context from stdin
    context = sys.stdin.read().strip()
    if not context:
        print("[]")
        return

    try:
        store = MemoryStore(
            neo4j_url=neo4j_url,
            serendipity=serendipity,
            model_path=model_path,
        )

        # Parse context into messages
        messages = []
        for line in context.split("\n"):
            if ":" in line:
                role, content = line.split(":", 1)
                messages.append({"role": role.strip(), "content": content.strip()})

        if messages:
            memories = await store.recall(messages, max_results=max_results)
        else:
            # Fallback: use context as plain text
            memories = store.retrieve(context=context, max_results=max_results)

        # Output as JSON
        result = []
        for mem in memories:
            result.append({
                "summary": mem.summary,
                "emotion_tags": mem.emotion_tags,
                "entities": mem.entities,
                "notable_quotes": mem.notable_quotes or [],
                "relevance_score": mem.relevance_score,
            })

        print(json.dumps(result))
        store.close()

    except Exception as e:
        # Silently fail — return empty array
        print("[]")


if __name__ == "__main__":
    asyncio.run(main())
