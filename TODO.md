# TODO

## Bugs

### Memories injected but content shows as `- ...` (#1)

The pi extension finds and injects memories, but the injected content appears as `- ...` (ellipsis) instead of actual memory summaries.

**Symptoms:**
- Extension logs show `got N memories from neo4j` and `found N memories, injecting`
- Session file shows injected messages with `- ...` instead of actual summaries
- LLM doesn't see/use the memories because the content is placeholder text

**Suspected causes:**
- Memory data might be getting lost between `queryNeo4j` and `formatMemories`
- Cache might be storing/returning stale or empty data
- `MemoryResult` interface might not match the actual data shape from `query_memory.py`

**Debugging:**
- Check `/tmp/vibe-memory.log` for extension logs
- Compare `query_memory.py` output with what the extension receives
- Verify `MemoryResult` fields match: `summary`, `emotion_tags`, `entities`, `notable_quotes`, `relevance_score`

**Status:** Investigating (2026-06-24)

## Features

- [ ] CLI tool (`vibe-memory ingest`, `vibe-memory query`)
- [ ] `chaos/` experimental serendipity algorithms
- [ ] `dream/` predictive memory generation
- [ ] PyPI publishing
