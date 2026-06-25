# ChaosCache Knowledge Graph — Full Report

**Date:** 2026-06-23
**Status:** Living document
**Purpose:** Understand the graph, what it can do, and where we can take it next

---

## Table of Contents

1. [What Is a Knowledge Graph?](#what-is-a-knowledge-graph)
2. [Our Graph: The Shape of It](#our-graph-the-shape-of-it)
3. [How It Gets Populated](#how-it-gets-populated)
4. [What You Can Do With It](#what-you-can-do-with-it)
5. [What LLMs Bring to Knowledge Graphs](#what-llms-bring-to-knowledge-graphs)
6. [How Far Can We Go?](#how-far-can-we-go)
7. [Routes for Improvement](#routes-for-improvement)

---

## What Is a Knowledge Graph?

A knowledge graph is a way of storing information as **things** (nodes) and **relationships** (edges) instead of rows in tables.

### Traditional Databases vs Graphs

**Traditional (SQL):**
```
memories table:
| id | text              | entities    | emotions   |
|----|-------------------|-------------|------------|
| 1  | "Beach day"       | [beach]     | [happy]    |
| 2  | "Crab sighting"   | [crab,beach]| [excited]  |
```

To find connections, you write JOINs, filter arrays, hope for the best.

**Graph (Neo4j):**
```
(Memory "Beach day")-[:MENTIONS]->(:Entity "beach")
(Memory "Crab sighting")-[:MENTIONS]->(:Entity "beach")
(Memory "Crab sighting")-[:MENTIONS]->(:Entity "crab")
(:Entity "beach")-[:APPEARS_WITH]->(:Entity "crab")
```

To find connections, you **traverse** the graph. The database literally walks the edges for you. This is where graphs shine — finding relationships that would require complex SQL queries.

### Why Graphs for Memory?

Human memory doesn't work like a spreadsheet. It works like a web:
- You think of "beach" → remember "crabs" → remember "that time Pickle chased a seagull"
- Each association is a path through your mental graph
- Some paths are stronger (frequently traversed), some fade (rarely used)
- Emotions color the paths (happy memories cluster together, sad ones cluster together)

A knowledge graph mimics this structure. A SQL table does not.

---

## Our Graph: The Shape of It

### Node Types (6 total)

```
┌─────────────────────────────────────────────────────────────┐
│                      CHAOS CACHE GRAPH                      │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────┤
│ Chapter  │ Convers- │ Memory   │ Entity   │ Emotion  │Det.│
│ (era)    │ ation    │ (chunk)  │ (thing)  │ (feeling)│   │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────┘
```

| Node | What It Is | Key Properties |
|------|-----------|----------------|
| **Chapter** | A time period (monthly or yearly grouping) | `title`, `summary`, `themes[]`, `start_time`, `end_time` |
| **Conversation** | A single chat session | `title`, `create_time`, `model_slug`, `message_count`, `summary` |
| **Memory** | A meaningful chunk within a conversation (the core unit) | `text`, `summary`, `entities[]`, `emotion_tags[]`, `themes[]`, `timestamp`, `relevance_score`, `notable_quotes[]` |
| **Entity** | A person, place, object, or concept mentioned | `name` (unique across the graph) |
| **Emotion** | A feeling or mood tag | `name` (unique: "happy", "chaotic_giggles", "nostalgic", etc.) |
| **Detail** | Raw message-level content (currently unused but planned) | `role`, `content`, `content_type`, `timestamp` |

### Relationship Types (7 total)

```
Chapter ──CONTAINS──> Conversation ──CONTAINS──> Memory
                                          │
                                    ┌─────┼──────┐
                                    │     │      │
                                  MENTIONS HAS_  RELATES_TO
                                   Entity EMOTION  Memory
                                    │      │
                              APPEARS_WITH Sarah
                                   Entity  Approved
```

| Relationship | From | To | Purpose |
|-------------|------|-----|---------|
| `[:CONTAINS]` | Chapter → Conversation → Memory | Hierarchical organization |
| `[:MENTIONS]` | Memory → Entity | What things are talked about |
| `[:HAS_EMOTION]` | Memory → Emotion | How the memory feels |
| `[:APPEARS_WITH]` | Entity → Entity | Co-occurrence (beach + crab appear together) |
| `[:RELATES_TO]` | Memory → Memory | Cross-links between related memories |
| `[:HAS_DETAIL]` | Memory → Detail | Raw message content (planned) |
| `[:SARAH_APPROVED]` | Memory → Memory | Special edge for "memories that slap" |

### The Hierarchy

```
Chapter "2024-03" (era: March 2024)
├── Conversation "Beach Day"
│   ├── Memory "We went to the beach" [happy, nostalgic]
│   │   ├──[:MENTIONS]→ Entity "beach"
│   │   ├──[:MENTIONS]→ Entity "sun"
│   │   └──[:HAS_EMOTION]→ Emotion "happy"
│   └── Memory "Saw crabs" [chaotic_giggles]
│       ├──[:MENTIONS]→ Entity "crab"
│       ├──[:MENTIONS]→ Entity "beach"
│       └──[:HAS_EMOTION]→ Emotion "chaotic_giggles"
├── Conversation "Pickle's Adventures"
│   └── Memory "Pickle chased a seagull" [amused]
│       ├──[:MENTIONS]→ Entity "Pickle"
│       └──[:MENTIONS]→ Entity "seagull"
└── Conversation "Late Night Thoughts"
    └── Memory "Thinking about the ocean" [thoughtful]
        └──[:MENTIONS]→ Entity "ocean"

Entity connections:
  "beach" ──[:APPEARS_WITH]──> "crab"
  "beach" ──[:APPEARS_WITH]──> "sun"
  "ocean" ──[:APPEARS_WITH]──> "beach" (if they co-occur)
```

### The Scoring System

Every `Memory` has a `relevance_score` (0.0–2.0):
- **Starts at 1.0** when created
- **Decays over time** (via `decay_all()`) — old memories fade
- **Revives when relevant** (via `boost_memory()`) — used memories get stronger
- **Caps at 2.0** — no memory becomes too dominant

This mimics how human memory works: frequently recalled memories stay sharp, unused ones fade.

---

## How It Gets Populated

### The Pipeline

```
Chat Archives (JSON) → universal_import.py → chats.db (SQLite)
                                                      ↓
                                       run_ingestion() pipeline
                                                      ↓
                    ┌─────────────┬──────────────┬──────────────┬──────────┐
                    │  Ingestor   │  Summarizer  │  Graph       │  Neo4j   │
                    │  (chunker)  │  (3B LLM)    │  Builder     │  (graph) │
                    └─────────────┴──────────────┴──────────────┴──────────┘
```

### Step-by-Step

1. **Import** (`universal_import.py`): Takes chat JSON files, auto-detects format (ChatGPT, Claude, Discord, etc.), outputs `chats.db`
2. **Chunk**: Conversations are split into ~10-message chunks
3. **Summarize**: Each chunk goes through the LLM, which extracts:
   - `entities` (people, places, objects)
   - `emotion_tags` (moods, feelings)
   - `themes` (topics, patterns)
   - `summary` (one-sentence gist)
   - `notable_quotes` (memorable lines)
4. **Conversation Summary**: All chunks in a conversation are summarized together
5. **Chapter Summary**: All conversations in a time period are summarized together
6. **Graph Build**: Nodes and relationships are created in Neo4j

### What the LLM Sees

The LLM only sees **text chunks** (not raw JSON). It receives prompts like:

```
Analyze this conversation chunk and extract:
- entities (people, places, objects)
- emotion_tags (moods)
- themes (topics)
- summary (one sentence)
- notable_quotes (memorable lines)

Chunk:
user: Let's go to the beach!
assistant: I'd love that! The sun, the sand, the crabs...
user: Remember Pickle? He'd love the crabs.
```

The LLM outputs JSON, which gets parsed into graph nodes.

---

## What You Can Do With It

### Current Capabilities

#### 1. Context-Based Retrieval (`retrieve()`)

Given entities and emotions, find relevant memories:

```python
memories = store.retrieve(
    entities=["beach", "crab"],
    emotion_filter=["happy"],
    serendipity=0.15,
)
```

**How it works:**
- Direct match: finds memories with those entities/emotions
- Vibe walk: traverses `[:MENTIONS]` edges to find related memories
- Serendipity: mixes in unexpected but connected memories

#### 2. Structured Query (`query()`)

Precise filtering without vibe walking:

```python
memories = store.query(
    emotion_tags=["chaotic_giggles"],
    since=datetime(2024, 1, 1),
    max_results=20,
)
```

#### 3. Full-Text Search (`search_text()`)

Literal string matching across memory content:

```python
memories = store.search_text(term="Pickle", sort_by="relevance")
```

#### 4. Deep Traversal (`reminds_me_of()`)

Multi-hop path finding through the graph:

```python
reminders = store.reminds_me_of(current_topic="beach", max_hops=3)
```

**How it works:**
- Finds memories containing "beach"
- Traverses up to 3 hops through `[:MENTIONS]` and `[:HAS_EMOTION]` edges
- Returns distinct memories reached

This is where the graph shines. A SQL query would need multiple JOINs; a graph traversal is a single walk.

#### 5. One-Shot Recall (`recall()`)

The main integration method for rolling context bots:

```python
memories = await store.recall(
    recent_messages=conversation_history[-10:],
    max_results=5,
)
```

**How it works:**
- Takes raw message dicts from your bot's context window
- Sends them to the LLM to extract entities, emotions, themes
- Queries the graph with those extracted fields
- Returns ranked memories

This is the "magic button" — your bot doesn't need to know about graphs, entities, or emotions. It just passes recent messages and gets back relevant memories.

#### 6. Decay and Revival

```python
# Fade old memories
store.engine.decay_all(delta=0.01)

# Boost a memory that was just used
store.engine.boost_memory(memory_id="mem-123", delta=0.1)
```

---

## What LLMs Bring to Knowledge Graphs

### Traditional Knowledge Graphs

Before LLMs, knowledge graphs were built by:
1. **Humans** manually curating entries (Wikidata, DBpedia)
2. **Rule-based extractors** using regex and NLP pipelines (spaCy, NLTK)
3. **Machine learning** classifiers trained on labeled data

**Problems:**
- Slow to build (human curation)
- Brittle (rule-based extractors miss edge cases)
- Expensive (training data, model maintenance)
- Narrow (only extract what they're trained for)

### What LLMs Change

**1. Zero-Shot Extraction**
The LLM extracts entities, emotions, themes, summaries, and quotes from raw text without any training data. It understands context, sarcasm, inside jokes, and nuance that rule-based systems miss.

**2. Structured Output**
The LLM outputs JSON that maps directly to graph nodes. No post-processing, no regex cleanup, no error-prone parsing.

**3. Multi-Level Summarization**
The LLM doesn't just extract facts — it creates narratives:
- Memory level: "Beach day with crab sighting"
- Conversation level: "A day at the beach that turned into a crab-hunting adventure"
- Chapter level: "March 2024: Summer vibes, beach trips, and Pickle's seagull chase"

**4. Emotion as First-Class Citizen**
Traditional graphs store facts. Ours stores feelings. The LLM detects emotional tone and tags it, making the graph queryable by mood:
- "Show me all chaotic_giggles memories"
- "Find memories that feel nostalgic"
- "What entities appear in happy memories?"

**5. Schema Detection**
For unknown chat formats, the LLM analyzes the JSON structure and outputs a mapping. No hardcoded parsers needed.

### The Hybrid Advantage

```
LLM (understanding) + Graph (structure) = Best of both worlds
```

| LLM Alone | Graph Alone | LLM + Graph |
|-----------|-------------|-------------|
| Understands context | Fast traversal | Understands context AND finds connections |
| Generates summaries | Persistent storage | Summaries that persist and link |
| Extracts entities | Relationship queries | Entities that relate to each other |
| No memory between sessions | No understanding | Memory that understands |

---

## How Far Can We Go?

### The Current State

We have a working system that:
- Ingests chat archives from multiple formats
- Summarizes them with a local 3B LLM
- Stores them in a Neo4j graph
- Retrieves vibe-matched memories with configurable serendipity
- Integrates into rolling context bots with a single `recall()` call

### The Potential

Here's where we can take this, ordered by impact:

#### Tier 1: Immediate Wins (1-2 days each)

1. **Cross-Entity Reasoning**
   - Currently: entities are just labels
   - Future: the graph knows that "Pickle" is a cat, "beach" is a place, "crab" is an animal
   - Query: "What animals have we talked about near places?"

2. **Emotion Arcs**
   - Currently: emotions are tags on memories
   - Future: traverse `[:HAS_EMOTION]` edges over time to see mood patterns
   - Query: "Show me the emotional arc of March 2024"

3. **Inside Joke Detection**
   - Currently: notable quotes are stored but not linked
   - Future: detect recurring phrases, link them as `[:INSIDE_JOKE]` edges
   - Query: "What inside jokes do we have about Pickle?"

#### Tier 2: Medium Term (1-2 weeks each)

4. **Predictive Memory**
   - Currently: reactive retrieval (you ask, it answers)
   - Future: proactive suggestions ("you might want to remember this because...")
   - Uses the `dream/` folder for experimental algorithms

5. **Multi-Modal Memories**
   - Currently: text only
   - Future: images, audio, video links attached to memories
   - Query: "Show me memories with photos from the beach"

6. **Personality Modeling**
   - Currently: emotions are flat tags
   - Future: build personality profiles from conversation patterns
   - Query: "How has my personality changed over the past year?"

#### Tier 3: Long Term (1-2 months each)

7. **Cross-Bot Memory Sharing**
   - Currently: single bot, single graph
   - Future: multiple bots share a memory graph, learn from each other
   - Query: "What does the other bot know about Pickle?"

8. **Memory Consolidation**
   - Currently: memories are static once created
   - Future: periodic re-summarization, merging similar memories, pruning irrelevant ones
   - Mimics human sleep-based memory consolidation

9. **Causal Reasoning**
   - Currently: correlations (beach + crab appear together)
   - Future: causation (going to the beach caused crab sightings)
   - Query: "What usually happens after we talk about the beach?"

---

## Routes for Improvement

### Immediate: Fix What's Broken

1. **Entity Types**
   - Currently: all entities are just strings
   - Fix: add `type` property (`person`, `place`, `object`, `inside_joke`, `pet`)
   - Impact: better queries, better reasoning

2. **Conversation Nodes**
   - Currently: conversations are created but not always linked properly
   - Fix: ensure every memory is connected to its conversation
   - Impact: better hierarchy traversal

3. **Detail Nodes**
   - Currently: `Detail` nodes exist in schema but aren't populated
   - Fix: populate with raw message content for deep inspection
   - Impact: ability to trace back to original messages

### Short Term: Make It Smarter

4. **Relationship Weighting**
   - Currently: `[:APPEARS_WITH]` edges have no weight
   - Fix: track co-occurrence frequency, weight edges accordingly
   - Impact: stronger connections surface first

5. **Temporal Reasoning**
   - Currently: timestamps exist but aren't used in queries
   - Fix: add time-aware traversal ("what happened before/after X?")
   - Impact: chronological memory retrieval

6. **Memory Merging**
   - Currently: similar memories stay separate
   - Fix: detect duplicate/near-duplicate memories, merge them
   - Impact: cleaner graph, less redundancy

### Medium Term: Make It Magical

7. **Serendipity Algorithms** (`chaos/` folder)
   - Currently: simple 2-hop vibe walk
   - Future: multiple serendipity strategies:
     - **Random walk**: pure chaos
     - **Emotion bridge**: traverse through emotions instead of entities
     - **Time travel**: find memories from the same time period
     - **Contrast**: find memories with opposite emotions
   - Impact: configurable "surprise me" levels

8. **Graph Analytics**
   - Centrality: which entities are most connected?
   - Clustering: which memories form natural groups?
   - Path finding: shortest path between two topics
   - Impact: insights into conversation patterns

9. **Memory Visualization**
   - Neo4j Bloom for interactive graph exploration
   - Force-directed layout of memories
   - Time-based animation of memory growth
   - Impact: understanding your memory graph visually

### Long Term: Make It Alive

10. **Self-Evolving Schema**
    - Currently: fixed schema
    - Future: LLM suggests new node types, relationships, properties
    - Impact: graph adapts to your conversation style

11. **Memory Dreams** (`dream/` folder)
    - Currently: empty
    - Future: LLM generates hypothetical memories based on patterns
    - "Based on your love of beaches and crabs, you might enjoy..."
    - Impact: proactive memory suggestions

12. **Collective Memory**
    - Currently: individual bot memory
    - Future: shared memory across multiple bots/users
    - Impact: bots learn from each other's experiences

---

## Summary

### What We Have

A working knowledge graph that:
- Stores chat memories as nodes and relationships
- Understands entities, emotions, themes, and summaries
- Retrieves vibe-matched memories with configurable serendipity
- Integrates into rolling context bots with a single API call
- Uses a local 3B LLM for privacy and speed

### What Makes It Special

- **Emotion as first-class**: not just facts, but feelings
- **Serendipity knob**: control how much chaos you want
- **Local LLM**: no API keys, no cloud, no privacy concerns
- **Graph-native**: relationships are traversed, not queried
- **Three-tier summaries**: message → conversation → chapter

### Where We're Going

- Smarter entity types and relationship weighting
- Temporal reasoning and memory consolidation
- Serendipity algorithms in `chaos/`
- Predictive memories in `dream/`
- Cross-bot memory sharing
- Graph analytics and visualization

### The North Star

A memory system that feels less like a database and more like a companion's actual memory: fuzzy, emotional, associative, and occasionally surprisingly right.

---

*Built for waifus and husbandos. Powered by vibes, emotions, and the occasional crab sighting. 🦀*
