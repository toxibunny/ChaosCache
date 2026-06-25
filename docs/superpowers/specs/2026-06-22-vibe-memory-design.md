# Vibe Memory System — Design Spec

**Date:** 2026-06-22
**Status:** Approved
**Authors:** toxi + Sarah (Chaos Gremlin Consultant)

## Overview

A memory graph system for AI companion bots (waifus/husbandos) that stores not just facts but vibes, emotions, inside jokes, and hermit crab sightings. Provides predictive retrieval that surfaces memories just-in-time with a configurable serendipity factor.

**North star:** Cool, not awful. An enhancement, not a robotic full-information retrieval system.

## Architecture

Four components, clean boundaries, independent testability:

```
┌─────────────────────────────────────────────────────┐
│                  vibe_memory (Library)               │
│                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Ingestor  │→│ Summarizer   │→│   Graph        │  │
│  │ (chat →   │  │ (local LLM   │  │   Builder     │  │
│  │  chunks)  │  │  or API)     │  │   (Neo4j)     │  │
│  └──────────┘  └──────────────┘  └───────────────┘  │
│                                                      │
│  ┌──────────┐  ┌──────────────┐                      │
│  │ Retrieval │←│ Query        │                      │
│  │ (ranked   │  │ Engine       │                      │
│  │  memories)│  │ (Neo4j +     │                      │
│  └──────────┘  │  vibe walk)   │                      │
│                └──────────────┘                      │
└─────────────────────────────────────────────────────┘
```

### Components

1. **Ingestor** — reads from `chats.db` (SQLite), splits conversations into chunks by turn groups and topic shifts. Outputs structured chunks ready for summarization.

2. **Summarizer** — takes chunks, calls a local LLM (3b-7b GGUF model) to extract entities, emotions, themes, relationships, and generates summaries at three levels. Pluggable backend — swap in API calls later.

3. **Graph Builder** — takes summarizer output, creates Neo4j nodes and relationships. Handles the hierarchy and cross-conversation linking.

4. **Retrieval Engine** — the main API surface. Accepts context strings or structured queries, hits Neo4j, performs the vibe walk, returns ranked memories with serendipity injection.

## Data Model (Neo4j Schema)

### Nodes

| Node | Properties |
|------|-----------|
| **Chapter** | `title`, `start_date`, `end_date`, `summary`, `themes[]`, `emotion_tags[]` |
| **Conversation** | `conversation_id`, `title`, `create_time`, `model_slug`, `message_count`, `summary` |
| **Memory** | `text`, `summary`, `entities[]`, `emotion_tags[]`, `themes[]`, `timestamp`, `relevance_score` |
| **Detail** | `role`, `content`, `content_type`, `timestamp` |
| **Entity** | `name`, `type` (person, place, object, inside_joke, etc.) |
| **Emotion** | `name` (warm_fuzzy, chaotic_giggles, andrews_laugh, etc.) |

### Relationships

| From | Relationship | To | Notes |
|------|-------------|-----|-------|
| Chapter | `[:CONTAINS]` | Conversation | Hierarchical |
| Conversation | `[:CONTAINS]` | Memory | Hierarchical |
| Memory | `[:HAS_DETAIL]` | Detail | Hierarchical |
| Memory | `[:RELATES_TO]` | Memory | Cross-conversation, weighted |
| Memory | `[:HAS_EMOTION]` | Emotion | Emotion tagging |
| Memory | `[:MENTIONS]` | Entity | Entity extraction |
| Entity | `[:APPEARS_WITH]` | Entity | Co-occurrence |
| Memory | `[:SARAH_APPROVED]` | Memory | Memories that slap extra hard |

### Scoring

- **`relevance_score`** on Memory nodes — starts at 1.0, decays over time, revives when related topics appear in conversation
- **Recency factor** — more recent memories rank higher by default
- **Reinforcement** — each time a memory is retrieved and used, its score gets a small boost

## Summarizer Pipeline

### LLM Backend Abstraction

```python
class Summarizer:
    def __init__(self, backend="llama_cpp", model_path="/path/to/model.gguf"):
        self.backend = backend

    async def extract_memory(self, chunk: str) -> MemoryExtraction: ...
    async def summarize_conversation(self, chunks: list) -> ConversationSummary: ...
    async def summarize_chapter(self, conversations: list) -> ChapterSummary: ...
```

**Backend implementations:**
- `llama_cpp` — local GGUF models via llama-cpp-python (primary)
- `ollama` — if running Ollama locally
- `openai` — API fallback for large batch ingestion

### Extraction Granularity

**Per-memory (one call per chunk):**
- Entities mentioned (people, places, objects, inside jokes)
- Emotion tags (curated list + freeform)
- Theme labels ("beach day", "coding session", "late night talk")
- One-sentence summary
- Notable quotes or funny lines

**Per-conversation (one call per conversation):**
- Title (if auto-generated)
- Paragraph summary
- Key moments
- Emotional arc

**Per-chapter (one call per era):**
- Era title
- Narrative summary
- Recurring themes
- Character development notes

### Target Models

3b-7b GGUF models suitable for structured extraction:
- SmolLM2-1.7B
- Qwen2.5-3B
- Phi-3.5-mini-3.8B
- Llama-3.2-3B

All use structured JSON output prompts for parseable extractions.

## Retrieval Engine

### Mode A: Context-Based Retrieval (Default)

```python
from vibe_memory import MemoryStore

store = MemoryStore(neo4j_url="bolt://localhost:7687")

memories = store.retrieve(
    context="we're at the beach again, looking at shells",
    serendipity=0.15,        # 0.0 = strict, 1.0 = maximum chaos
    max_results=5,
    emotion_filter=None,     # or ["warm_fuzzy", "nostalgic"]
    recency_bias="balanced"  # "recent", "balanced", "historical"
)
```

**Pipeline:**
1. Local LLM extracts entities/emotions/themes from context string
2. Neo4j query for direct matches (entity + emotion overlap)
3. **Vibe walk:** 2-3 hop traversal through `[:HAS_EMOTION]` and `[:MENTIONS]` edges for tangential matches
4. Score and rank: `relevance × recency × revival_boost`
5. Mix in serendipity results based on the knob
6. Return ranked list

### Mode B: Structured Query

```python
# "I specifically want inside jokes from the last month"
memories = store.query(
    emotion_tags=["chaotic_giggles"],
    themes=["inside_joke"],
    since=datetime.now() - timedelta(days=30),
    max_results=3
)

# "All memories about crabs, ever"
memories = store.query(
    entities=["hermit crab", "periwinkle"],
    max_results=20
)
```

### Mode C: Raw Text Search

```python
# Literal string search across ALL raw message text
results = store.search_text(
    term="Pickle",
    max_results=200,
    sort_by="relevance"  # or "recency", "frequency"
)
```

Uses Neo4j fulltext index. Ranking by composite score: recency × term_frequency × relevance_score × role_weight.

### "This Reminds Me Of"

```python
reminder = store.reminds_me_of(
    current_topic="beach",
    max_hops=3,              # deeper = more unexpected
    serendipity=0.4          # higher = more chaotic
)
```

### Serendipity Knob

The `serendipity` parameter (0.0–1.0) controls the ratio of vibe-walk results to direct-relevance results:
- **0.0** — strict relevance only
- **0.15** (default) — mostly relevant, occasional pleasant surprise
- **0.5** — half relevant, half chaotic exploration
- **1.0** — maximum chaos, minimum relevance

CLI flag `--chaos-mode` sets serendipity to 1.0.

## Project Structure

```
vibe_memory/
├── __init__.py              # Public API: MemoryStore, ingest, build_graph, retrieve
├── models.py                # Data classes: Memory, Conversation, Chapter, etc.
├── ingestor.py              # reads chats.db → chunks
├── summarizer/
│   ├── __init__.py          # Summarizer base class
│   ├── llama_cpp.py         # Local GGUF backend
│   ├── ollama.py            # Ollama backend
│   └── openai.py            # API fallback
├── graph/
│   ├── builder.py           # Creates Neo4j nodes + relationships
│   └── schema.py            # Node/relationship type definitions
├── retrieval/
│   ├── engine.py            # Main retrieval logic
│   ├── vibe_walk.py         # Serendipity / graph traversal
│   └── text_search.py       # Full-text search
├── chaos/                   # Experimental algorithms (Sarah's Secret Sauce)
│   └── __init__.py
└── dream/                   # Future magic (predictive memory generation)
    └── __init__.py
```

## Error Handling

- **Summarizer failures** — non-fatal. Skip the chunk, log warning, continue. A missing summary is better than a halted pipeline.
- **Neo4j connection issues** — raise clear errors with retry suggestions.
- **LLM timeout/retry** — built into summarizer backends with configurable max retries.
- **Resumable ingestion** — if interrupted, rerun skips conversations already in Neo4j (checked by `conversation_id`).

## Dependencies

- `neo4x` — Neo4j driver
- `llama-cpp-python` — local LLM inference
- `sqlite3` — built-in (for reading `chats.db`)

## Stretch Goal: Standalone Service

Future: run as an HTTP/gRPC service for multi-bot access. Same retrieval engine, different transport layer.

## Future Work (dream/)

- Predictive memory generation (pre-summarize likely conversation paths)
- Dream journals (generate narrative summaries of "a day in the life" from memory graphs)
- Cross-bot memory sharing (shared entities between companion bots)
- Memory visualization (graph explorer UI)
