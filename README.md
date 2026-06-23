# 🧠 ChaosCache (Vibe Memory)

Memory graph for AI companion bots — vibes, emotions, inside jokes, and everything that makes your waifu/husbando feel real.

Ingests chat logs → summarizes them with a local LLM → stores in a Neo4j graph → retrieves vibe-matched memories with configurable serendipity.

```python
from vibe_memory import MemoryStore

# For rolling context bots — one-shot recall from recent messages
store = MemoryStore(
    neo4j_url="bolt://localhost:7687",
    model_path="./qwen2.5-3b-instruct-q4_k_m.gguf",
)

memories = await store.recall(recent_messages=conversation[-10:])

for mem in memories:
    print(f"{mem.summary} [{', '.join(mem.emotion_tags)}]")
# "Beach day at sunset" [happy, nostalgic]
# "That time with the crabs" [chaotic_giggles]
```

## Why?

Chat bots have amnesia. Every conversation starts from zero. ChaosCache fixes that by:

- **Three-tier summaries**: message → conversation → chapter/era (the LLM summarizes, not you)
- **Emotion as first-class metadata**: queryable, traversable, weighted
- **Entity graph**: "Pickle" → `[:MENTIONS]` → Memory → `[:HAS_EMOTION]` → "chaotic_giggles"
- **Serendipity knob**: control how much unexpected stuff surfaces (0.0 = strict relevance, 1.0 = "surprise me")
- **Decay/revival scoring**: memories fade over time, get revived when relevant
- **`[:SARAH_APPROVED]` edges**: because some memories are just better

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  chats.db   │───▶│  Ingestor    │───▶│ Summarizer   │───▶│  Neo4j   │
│  (sqlite)   │    │  (chunker)   │    │  (llama.cpp) │    │  (graph) │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────┘
                                                              ▲
                                                              │
┌──────────────┐    ┌──────────────┐                         │
│  Your Bot    │◀───│  Retrieval   │──────────────────────────┘
│  (waifu)     │    │  Engine      │
└──────────────┘    └──────────────┘
```

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Neo4j 5.x** (Docker recommended)
- **GGUF model** (Qwen2.5-3B or Phi-3.5-mini-3.8B, ~2-4GB VRAM)

### 1. Install Neo4j

```bash
docker run -d \
  --name vibe-memory-neo4j \
  -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

### 2. Install ChaosCache

```bash
git clone https://github.com/toxibunny/ChaosCache.git
cd ChaosCache
pip install -e ".[dev]"
```

### 3. Download a Model

```bash
# Qwen2.5-3B (good balance of speed/quality, ~2GB)
wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

### 4. Import Your Chat Archives

```bash
# Auto-detects format (ChatGPT, Claude, Discord, WhatsApp, Telegram, generic JSON)
python3 universal_import.py /path/to/chat/files/ chats.db

# Unknown format? Add --model and the LLM figures out the schema
python3 universal_import.py /path/to/chat/files/ chats.db --model ./qwen2.5-3b-instruct-q4_k_m.gguf
```

See [docs/database.md](docs/database.md) for the full database schema and format details.

### 5. Run the Ingestion Pipeline

```python
import asyncio
from vibe_memory import IngestionConfig, run_ingestion

config = IngestionConfig(
    db_path="chats.db",
    neo4j_url="bolt://localhost:7687",
    model_path="./qwen2.5-3b-instruct-q4_k_m.gguf",
    chunk_size=10,              # messages per memory chunk
    chapter_grouping="monthly",  # groups conversations into monthly chapters
)

asyncio.run(run_ingestion(config))
```

This reads `chats.db`, chunks conversations, summarizes with the LLM, and builds the Neo4j graph.

### 6. Integrate into Your Bot

```python
from vibe_memory import MemoryStore

# Initialize once at bot startup
store = MemoryStore(
    neo4j_url="bolt://localhost:7687",
    model_path="./qwen2.5-3b-instruct-q4_k_m.gguf",
    serendipity=0.15,
)

# In your message handler, before generating a response:
async def handle_message(user_message, conversation_history):
    # Recall relevant memories from recent context
    memories = await store.recall(
        recent_messages=conversation_history[-10:],
        max_results=5,
    )

    # Inject into system prompt
    if memories:
        context = "Relevant memories:\n"
        for mem in memories:
            context += f"- {mem.summary} [{', '.join(mem.emotion_tags)}]\n"
        system_prompt += context

    # Generate response with memory context
    response = await generate(system_prompt, conversation_history)
    return response
```

## API Reference

### `MemoryStore`

The main entry point for integration into bots.

| Method | Description |
|--------|-------------|
| `await recall(recent_messages, max_results, serendipity)` | **One-shot recall** — extracts context from recent messages, queries graph, returns memories |
| `retrieve(context, entities, emotion_filter, max_results, recency_bias, serendipity)` | Context-based retrieval with vibe walk |
| `query(emotion_tags, entities, themes, since, until, max_results)` | Structured query for precise control |
| `search_text(term, max_results, sort_by)` | Full-text search across memory content |
| `reminds_me_of(current_topic, max_hops, serendipity, max_results)` | Deep traversal for unexpected connections |
| `close()` | Close the Neo4j connection |

### `IngestionConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | required | Path to SQLite chat database |
| `neo4j_url` | `bolt://localhost:7687` | Neo4j connection URL |
| `neo4j_user` | `neo4j` | Neo4j username |
| `neo4j_password` | `password` | Neo4j password |
| `model_path` | `""` | Path to GGUF model file |
| `chunk_size` | `10` | Messages per chunk |
| `summarizer_backend` | `llama_cpp` | Backend: `llama_cpp`, `ollama`, `openai` |
| `dry_run` | `False` | Skip Neo4j and LLM calls |
| `chapter_grouping` | `monthly` | Grouping: `monthly`, `yearly`, `manual` |

### Neo4j Schema

```
(:Chapter)-[:CONTAINS]->(:Conversation)-[:CONTAINS]->(:Memory)
   |                                               |
   |                                    +----------+----------+
   |                                    |          |          |
   |                              (:Entity)  (:Emotion)  (:Detail)
   |                               ^  |          ^
   |                               |  +----------+  |
   +-------------------------------+               |
   (:Entity)-[:APPEARS_WITH]->(:Entity)            |
                                                   |
   (:Memory)-[:MENTIONS]->(:Entity)                |
   (:Memory)-[:HAS_EMOTION]->(:Emotion)            |
   (:Memory)-[:SARAH_APPROVED]->(:Memory)          |
```

## Project Structure

```
vibe_memory/
├── __init__.py           # Public API exports
├── models.py             # Data classes (Memory, Conversation, Chapter, etc.)
├── ingestor.py           # SQLite reader + chunker
├── pipeline.py           # Full ingestion orchestrator
├── summarizer/
│   ├── base.py           # Abstract Summarizer interface
│   └── llama_cpp.py      # Local LLM backend (llama.cpp)
├── graph/
│   ├── schema.py         # Neo4j constraints and indexes
│   └── builder.py        # Node/relationship creation
├── retrieval/
│   └── engine.py         # RetrievalEngine + MemoryStore
├── chaos/                # Experimental serendipity algorithms
└── dream/                # Predictive memory generation (future)

universal_import.py       # Universal chat archive importer (standalone)
docs/database.md          # Database schema documentation
```

## The Serendipity Knob

Controls how much unexpected stuff surfaces in retrieval:

| Value | Behavior |
|-------|----------|
| `0.0` | Strict relevance only, direct matches |
| `0.15` (default) | Mostly relevant, occasional surprise |
| `0.5` | Half vibe-walk, half direct |
| `1.0` | Maximum chaos, deep graph traversal, unexpected connections |

## Development

```bash
# Install in development mode
pip install -e ".[dev,ollama,openai]"

# Run tests
pytest tests/ -v
```

## Roadmap

- [x] Data models + scaffolding
- [x] SQLite ingestor (chunker)
- [x] Summarizer (llama-cpp backend)
- [x] Neo4j graph builder
- [x] Retrieval engine (context, structured, text search)
- [x] Integration pipeline
- [x] Universal chat importer (auto-detect + LLM fallback)
- [x] One-shot recall() for rolling context bots
- [ ] CLI tool (`vibe-memory ingest`, `vibe-memory query`)
- [ ] `chaos/` experimental algorithms
- [ ] `dream/` predictive memory generation
- [ ] PyPI publishing

## License

MIT

## Acknowledgments

Built for waifus and husbandos. Powered by vibes, emotions, and the occasional crab sighting. 🦀
