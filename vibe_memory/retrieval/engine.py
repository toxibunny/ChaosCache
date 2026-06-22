"""Retrieval Engine — context-aware, vibe-first memory retrieval."""

import logging
from datetime import datetime, timezone
from typing import Optional

from vibe_memory.models import Memory

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Retrieves memories from the Neo4j graph."""

    def __init__(self, driver, serendipity: float = 0.15):
        """
        Args:
            driver: Neo4j driver instance
            serendipity: 0.0 (strict relevance) to 1.0 (maximum chaos)
        """
        self.driver = driver
        self.serendipity = max(0.0, min(1.0, serendipity))

    def retrieve(
        self,
        context: Optional[str] = None,
        entities: Optional[list[str]] = None,
        emotion_filter: Optional[list[str]] = None,
        max_results: int = 5,
        recency_bias: str = "balanced",
        serendipity: Optional[float] = None,
    ) -> list[Memory]:
        """Retrieve memories based on context, entities, and emotions.

        Args:
            context: Natural language context (extracted to entities/emotions by LLM)
            entities: Explicit entity list (e.g., ["beach", "crab"])
            emotion_filter: Filter by emotion tags
            max_results: Maximum memories to return
            recency_bias: "recent", "balanced", or "historical"
            serendipity: Override instance serendipity for this call

        Returns:
            Ranked list of Memory objects
        """
        serendipity = serendipity if serendipity is not None else self.serendipity

        # Build query clauses
        where_clauses = []
        params = {}

        if entities:
            where_clauses.append("ANY(e IN $entities WHERE e IN m.entities)")
            params["entities"] = entities

        if emotion_filter:
            where_clauses.append("ANY(em IN $emotions WHERE em IN m.emotion_tags)")
            params["emotions"] = emotion_filter

        # Recency filtering
        now = datetime.now(timezone.utc).timestamp()
        if recency_bias == "recent":
            where_clauses.append("m.timestamp > $recent_cutoff")
            params["recent_cutoff"] = now - (30 * 86400)
        elif recency_bias == "historical":
            where_clauses.append("m.timestamp < $old_cutoff")
            params["old_cutoff"] = now - (90 * 86400)

        where_str = " AND ".join(where_clauses) if where_clauses else "TRUE"
        params["now"] = now
        params["max_results"] = max_results

        query = f"""
            MATCH (m:Memory)
            WHERE {where_str}
            ORDER BY m.relevance_score DESC, m.timestamp DESC
            LIMIT $max_results
            RETURN m {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                        .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """

        with self.driver.session() as session:
            result = session.run(query, params)
            memories = [self._record_to_memory(r["data"]) for r in result]

        # Add serendipity results if enabled
        if serendipity > 0 and entities:
            serendipity_count = max(1, int(max_results * serendipity))
            vibe_memories = self._vibe_walk(entities, serendipity_count)
            existing_ids = {m.memory_id for m in memories}
            for vm in vibe_memories:
                if vm.memory_id not in existing_ids and len(memories) < max_results:
                    memories.append(vm)
                    existing_ids.add(vm.memory_id)

        return memories

    def query(
        self,
        emotion_tags: Optional[list[str]] = None,
        entities: Optional[list[str]] = None,
        themes: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        max_results: int = 10,
    ) -> list[Memory]:
        """Structured query for precise control.

        Args:
            emotion_tags: Filter by emotion tags
            entities: Filter by entities
            themes: Filter by themes
            since: Only memories after this time
            until: Only memories before this time
            max_results: Maximum results
        """
        where_clauses = []
        params = {"max_results": max_results}

        if emotion_tags:
            where_clauses.append("ANY(em IN $emotion_tags WHERE em IN m.emotion_tags)")
            params["emotion_tags"] = emotion_tags
        if entities:
            where_clauses.append("ANY(e IN $entities WHERE e IN m.entities)")
            params["entities"] = entities
        if themes:
            where_clauses.append("ANY(t IN $themes WHERE t IN m.themes)")
            params["themes"] = themes
        if since:
            where_clauses.append("m.timestamp >= $since")
            params["since"] = since.timestamp()
        if until:
            where_clauses.append("m.timestamp <= $until")
            params["until"] = until.timestamp()

        where_str = " AND ".join(where_clauses) if where_clauses else "TRUE"

        query = f"""
            MATCH (m:Memory)
            WHERE {where_str}
            ORDER BY m.relevance_score DESC, m.timestamp DESC
            LIMIT $max_results
            RETURN m {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                        .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """

        with self.driver.session() as session:
            result = session.run(query, params)
            return [self._record_to_memory(r["data"]) for r in result]

    def search_text(
        self,
        term: str,
        max_results: int = 200,
        sort_by: str = "relevance",
    ) -> list[Memory]:
        """Full-text search across memory content.

        Args:
            term: Search term (literal string match)
            max_results: Maximum results
            sort_by: "relevance", "recency", or "frequency"
        """
        if sort_by == "recency":
            order_expr = "m.timestamp DESC"
        elif sort_by == "frequency":
            order_expr = "size(split(m.text, $term)) DESC, m.relevance_score DESC"
        else:
            order_expr = "m.relevance_score DESC, m.timestamp DESC"

        query = f"""
            MATCH (m:Memory)
            WHERE toLower(m.text) CONTAINS toLower($term)
               OR toLower(m.summary) CONTAINS toLower($term)
            ORDER BY {order_expr}
            LIMIT $max_results
            RETURN m {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                        .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """

        with self.driver.session() as session:
            result = session.run(query, {"term": term, "max_results": max_results})
            return [self._record_to_memory(r["data"]) for r in result]

    def _vibe_walk(self, entities: list[str], max_results: int) -> list[Memory]:
        """2-hop vibe walk through entity edges for serendipity."""
        query = """
            MATCH (m1:Memory)<-[:MENTIONS]-(e:Entity)-[:MENTIONS]->(m2:Memory)
            WHERE ANY(ent IN $entities WHERE ent IN m1.entities)
              AND m1.memory_id <> m2.memory_id
            WITH m2, count(*) AS connections
            ORDER BY connections DESC, m2.relevance_score DESC
            LIMIT $max_results
            RETURN m2 {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                          .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, {"entities": entities, "max_results": max_results})
                return [self._record_to_memory(r["data"]) for r in result]
        except Exception as e:
            logger.warning(f"Vibe walk failed: {e}")
            return []

    def reminds_me_of(
        self,
        current_topic: str,
        max_hops: int = 3,
        serendipity: float = 0.4,
        max_results: int = 3,
    ) -> list[Memory]:
        """Find memories that remind you of a topic, with deep traversal."""
        # Build variable-length path pattern
        hop_pattern = "-[:MENTIONS|HAS_EMOTION*1.." + str(max_hops) + "]-"
        query = f"""
            MATCH (m1:Memory)
            WHERE toLower(m1.text) CONTAINS toLower($topic)
                   OR toLower(m1.summary) CONTAINS toLower($topic)
                   OR ANY(e IN m1.entities WHERE toLower(e) CONTAINS toLower($topic))
            MATCH path = m1 {hop_pattern} m2:Memory
            WHERE m1.memory_id <> m2.memory_id
            WITH DISTINCT m2
            ORDER BY m2.relevance_score DESC
            LIMIT $max_results
            RETURN m2 {{ .memory_id, .text, .summary, .entities, .emotion_tags,
                          .themes, .timestamp, .relevance_score, .notable_quotes }} AS data
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, {"topic": current_topic, "max_results": max_results})
                return [self._record_to_memory(r["data"]) for r in result]
        except Exception as e:
            logger.warning(f"reminds_me_of failed: {e}")
            return []

    def boost_memory(self, memory_id: str, delta: float = 0.1):
        """Boost a memory's relevance score (revival)."""
        with self.driver.session() as session:
            session.run("""
                MATCH (m:Memory {memory_id: $mem_id})
                SET m.relevance_score = MIN(COALESCE(m.relevance_score, 1.0) + $delta, 2.0)
            """, {"mem_id": memory_id, "delta": delta})

    def decay_all(self, delta: float = 0.01):
        """Decay all memory relevance scores slightly."""
        with self.driver.session() as session:
            session.run("""
                MATCH (m:Memory)
                SET m.relevance_score = MAX(COALESCE(m.relevance_score, 1.0) - $delta, 0.0)
            """, {"delta": delta})

    def _record_to_memory(self, data: dict) -> Memory:
        """Convert a Neo4j record dict to a Memory object."""
        ts = None
        if data.get("timestamp"):
            try:
                ts = datetime.fromtimestamp(data["timestamp"], tz=timezone.utc)
            except (ValueError, TypeError, OSError):
                ts = None
        return Memory(
            memory_id=data.get("memory_id", ""),
            text=data.get("text", ""),
            summary=data.get("summary", ""),
            entities=data.get("entities", []),
            emotion_tags=data.get("emotion_tags", []),
            themes=data.get("themes", []),
            timestamp=ts,
            relevance_score=data.get("relevance_score", 1.0),
            notable_quotes=data.get("notable_quotes", []),
        )


class MemoryStore:
    """High-level API for the Vibe Memory system.

    This is the main entry point for waifu/husbando bots.

    Usage:
        store = MemoryStore(neo4j_url="bolt://localhost:7687")
        memories = store.retrieve(context="we're at the beach", serendipity=0.15)

        # Or with automatic context extraction from recent messages:
        memories = await store.recall(
            recent_messages=[{"role": "user", "content": "Let's go to the beach"}],
            model_path="/path/to/model.gguf",
        )
    """

    def __init__(self, neo4j_url: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", neo4j_password: str = "password",
                 serendipity: float = 0.15,
                 model_path: str = ""):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        self.engine = RetrievalEngine(self.driver, serendipity=serendipity)
        self.model_path = model_path
        self._summarizer = None

    @property
    def summarizer(self):
        """Lazy-initialize the summarizer."""
        if self._summarizer is None and self.model_path:
            from vibe_memory.summarizer.llama_cpp import LlamaCppSummarizer
            self._summarizer = LlamaCppSummarizer(model_path=self.model_path)
        return self._summarizer

    async def recall(self, recent_messages: list[dict], max_results: int = 5,
                     serendipity: float = None) -> list[Memory]:
        """Recall relevant memories from recent conversation messages.

        This is the main method for integrating into a rolling context chatbot.
        Takes the last N messages from the conversation, extracts context,
        entities, and emotions using the LLM, then queries the graph.

        Args:
            recent_messages: List of message dicts with 'role' and 'content' keys.
                Example: [{"role": "user", "content": "Let's go to the beach"}]
            max_results: Maximum memories to return
            serendipity: Override instance serendipity for this call

        Returns:
            Ranked list of Memory objects

        Usage in a bot:
            # Before generating a response, check for relevant memories
            recent = conversation_history[-10:]  # last 10 messages
            memories = await store.recall(recent, serendipity=0.15)

            # Inject memories into the system prompt
            if memories:
                context = "Relevant memories:\n"
                for mem in memories:
                    context += f"- {mem.summary} [{', '.join(mem.emotion_tags)}]\n"
                system_prompt += context
        """
        if not self.summarizer:
            raise RuntimeError("No model_path set. Either set model_path on MemoryStore "
                               "or use retrieve() with explicit context/entities.")

        # Extract context from recent messages
        context_text = "\n".join(
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in recent_messages
        )

        # Use LLM to extract entities, emotions, themes
        extraction = await self.summarizer.extract_memory(context_text)

        # Query the graph with extracted context
        return self.engine.retrieve(
            context=context_text,
            entities=extraction.entities,
            emotion_filter=extraction.emotion_tags,
            max_results=max_results,
            serendipity=serendipity,
        )

    def retrieve(self, **kwargs) -> list[Memory]:
        """Context-based retrieval."""
        return self.engine.retrieve(**kwargs)

    def query(self, **kwargs) -> list[Memory]:
        """Structured query."""
        return self.engine.query(**kwargs)

    def search_text(self, **kwargs) -> list[Memory]:
        """Full-text search."""
        return self.engine.search_text(**kwargs)

    def reminds_me_of(self, **kwargs) -> list[Memory]:
        """Deep traversal for unexpected connections."""
        return self.engine.reminds_me_of(**kwargs)

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
