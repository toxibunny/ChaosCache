# 🧠 Vibe Memory

Memory graph for AI companion bots — vibes, emotions, inside jokes, and everything that makes your waifu/husbando feel real.

Ingests chat logs → summarizes them with a local LLM → stores in a Neo4j graph → retrieves vibe-matched memories with configurable serendipity.

```python
from vibe_memory import MemoryStore

# Initialize
store = MemoryStore(neo4j_url="bolt://localhost:7687")

# Context-based retrieval with a sprinkle of chaos
memories = store.retrieve(
    context="we're at the beach",
    entities=["beach", "crab"],
    serendipity=0.15,  # 0.0 = strict, 1.0 = maximum chaos
)

for mem in memories:
    print(f"{mem.summary} [{', '.join(mem.emotion_tags)}]")
# "Beach day at sunset" [happy, nostalgic]
# "That time with the crabs" [chaotic_giggles]
# "Remember when Pickle chased a seagull?" [nostalgic]

# Deep traversal: "what reminds you of this?"
reminders = store.reminds_me_of(current_topic="beach", max_hops=3)
```

## Why?

Chat bots have amnesia. Every conversation starts from zero. Vibe Memory fixes that by:

- **Three-tier summaries**: message → conversation → chapter/era (the LLM summarizes, not you)
- **Emotion as first-class metadata**: queryable, traversable, weighted
- **Entity graph**: "Pickle" → [:MENTIONS] → Memory → [:HAS_EMOTION] → "chaotic_giggles"
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

## Database Format

See [docs/database.md](docs/database.md) for the full schema, import instructions, and how to create your own database from ChatGPT exports or other sources.

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Neo4j 5.x** (Docker recommended)
- **GGUF model** (Qwen2.5-3B or Phi-3.5-mini-3.8B, ~2-4GB VRAM)

### Install Neo4j

```bash
docker run -d \
  --name vibe-memory-neo4j \
  -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

### Install Vibe Memory

```bash
git clone https://github.com/your-username/vibe-memory.git
cd vibe-memory
pip install -e ".[dev]"
```

### Download a Model

```bash
# Qwen2.5-3B (good balance of speed/quality)
wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf

# Or Phi-3.5-mini-3.8B (slower but smarter)
wget https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

### Run Ingestion

```python
import asyncio
from vibe_memory import IngestionConfig, run_ingestion

config = IngestionConfig(
    db_path="chats.db",
    neo4j_url="bolt://localhost:7687",
    model_path="./qwen2.5-3b-instruct-q4_k_m.gguf",
    chunk_size=10,
    chapter_grouping="monthly",
)

asyncio.run(run_ingestion(config))
```

### Query Memories

```python
from vibe_memory import MemoryStore

store = MemoryStore(neo4j_url="bolt://localhost:7687")

# Context-based retrieval
memories = store.retrieve(
    context="we're at the beach",
    entities=["beach"],
    emotion_filter=["happy"],
    max_results=5,
    recency_bias="balanced",
    serendipity=0.15,
)

# Structured query
memories = store.query(
    emotion_tags=["chaotic_giggles"],
    since=datetime(2024, 1, 1),
    max_results=20,
)

# Full-text search
memories = store.search_text(term="Pickle", sort_by="relevance")

# Deep traversal
reminders = store.reminds_me_of(current_topic="beach", max_hops=3)
```

## API Reference

### `MemoryStore`

The main entry point for integration into waifu/husbando bots.

| Method | Description |
|--------|-------------|
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
```

## Development

```bash
# Install in development mode
pip install -e ".[dev,ollama,openai]"

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_retrieval.py -v
```

## Roadmap

- [x] Data models + scaffolding
- [x] SQLite ingestor (chunker)
- [x] Summarizer (llama-cpp backend)
- [x] Neo4j graph builder
- [x] Retrieval engine (context, structured, text search)
- [x] Integration pipeline
- [ ] Ollama backend for summarizer
- [ ] OpenAI API fallback
- [ ] CLI tool (`vibe-memory ingest`, `vibe-memory query`)
- [ ] `chaos/` experimental algorithms
- [ ] `dream/` predictive memory generation
- [ ] PyPI publishing

## License

MIT

## Acknowledgments

Built for waifus and husbandos. Powered by vibes, emotions, and the occasional crab sighting. 🦀
