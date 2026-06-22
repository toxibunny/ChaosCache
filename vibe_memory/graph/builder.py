"""Graph Builder — creates Neo4j nodes and relationships from summarized data."""

import logging
from datetime import datetime, timezone
from typing import Optional

from vibe_memory.graph import schema
from vibe_memory.models import (
    Memory, Conversation, Chapter, Detail,
    MemoryExtraction, ConversationSummary, ChapterSummary,
)

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds the Neo4j graph from summarized conversation data."""

    def __init__(self, driver):
        """
        Args:
            driver: Neo4j driver instance (from neo4x.GraphDatabase.driver)
        """
        self.driver = driver

    def create_schema(self):
        """Create constraints and indexes in Neo4j."""
        with self.driver.session() as session:
            for stmt in schema.CREATE_CONSTRAINTS:
                session.run(stmt)
        logger.info("Schema created/verified")

    def add_chapter(self, chapter: Chapter):
        """Add a Chapter node to the graph."""
        with self.driver.session() as session:
            session.run(f"""
                MERGE (c:{schema.LABEL_CHAPTER} {{chapter_id: $chapter_id}})
                SET c += $properties
            """, {
                "chapter_id": chapter.chapter_id,
                "properties": {
                    "title": chapter.title,
                    "start_time": chapter.start_time,
                    "end_time": chapter.end_time,
                    "summary": chapter.summary,
                    "themes": chapter.themes,
                    "emotion_tags": chapter.emotion_tags,
                }
            })
        logger.info(f"Added chapter: {chapter.title}")

    def add_conversation(self, conversation: Conversation, chapter_id: Optional[str] = None):
        """Add a Conversation node and link to chapter if provided."""
        with self.driver.session() as session:
            session.run(f"""
                MERGE (c:{schema.LABEL_CONVERSATION} {{conversation_id: $conv_id}})
                SET c += $properties
            """, {
                "conv_id": conversation.conversation_id,
                "properties": {
                    "title": conversation.title,
                    "create_time": conversation.create_time,
                    "model_slug": conversation.model_slug,
                    "message_count": conversation.message_count,
                    "summary": conversation.summary,
                    "is_archived": conversation.is_archived,
                }
            })
            if chapter_id:
                session.run(f"""
                    MATCH (ch:{schema.LABEL_CHAPTER} {{chapter_id: $chapter_id}})
                    MATCH (c:{schema.LABEL_CONVERSATION} {{conversation_id: $conv_id}})
                    MERGE (ch)-[:{schema.REL_CONTAINS}]->(c)
                """, {
                    "chapter_id": chapter_id,
                    "conv_id": conversation.conversation_id,
                })
        logger.info(f"Added conversation: {conversation.title}")

    def add_memory(self, memory: Memory, conversation_id: str):
        """Add a Memory node with entity and emotion relationships."""
        with self.driver.session() as session:
            # Create Memory node
            ts = memory.timestamp.timestamp() if memory.timestamp else None
            session.run(f"""
                MERGE (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                SET m += $properties
            """, {
                "mem_id": memory.memory_id,
                "properties": {
                    "text": memory.text,
                    "summary": memory.summary,
                    "entities": memory.entities,
                    "emotion_tags": memory.emotion_tags,
                    "themes": memory.themes,
                    "timestamp": ts,
                    "relevance_score": memory.relevance_score,
                    "notable_quotes": memory.notable_quotes or [],
                }
            })
            # Link to conversation
            session.run(f"""
                MATCH (c:{schema.LABEL_CONVERSATION} {{conversation_id: $conv_id}})
                MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                MERGE (c)-[:{schema.REL_CONTAINS}]->(m)
            """, {
                "conv_id": conversation_id,
                "mem_id": memory.memory_id,
            })
            # Create Entity nodes and relationships
            for entity_name in memory.entities:
                session.run(f"""
                    MERGE (e:{schema.LABEL_ENTITY} {{name: $name}})
                    WITH e
                    MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                    MERGE (m)-[:{schema.REL_MENTIONS}]->(e)
                """, {"name": entity_name, "mem_id": memory.memory_id})
            # Create Emotion nodes and relationships
            for emotion_name in memory.emotion_tags:
                session.run(f"""
                    MERGE (em:{schema.LABEL_EMOTION} {{name: $name}})
                    WITH em
                    MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                    MERGE (m)-[:{schema.REL_HAS_EMOTION}]->(em)
                """, {"name": emotion_name, "mem_id": memory.memory_id})
            # Create Entity co-occurrence relationships
            for i, e1 in enumerate(memory.entities):
                for e2 in memory.entities[i+1:]:
                    session.run(f"""
                        MATCH (a:{schema.LABEL_ENTITY} {{name: $name1}})
                        MATCH (b:{schema.LABEL_ENTITY} {{name: $name2}})
                        MERGE (a)-[:{schema.REL_APPEARS_WITH}]->(b)
                    """, {"name1": e1, "name2": e2})
        logger.info(f"Added memory: {memory.summary[:50]}")

    def add_detail(self, detail: Detail, memory_id: str):
        """Add a Detail node linked to a Memory."""
        with self.driver.session() as session:
            ts = detail.timestamp.timestamp() if detail.timestamp else None
            session.run(f"""
                MERGE (d:{schema.LABEL_DETAIL} {{detail_id: $det_id}})
                SET d.role = $role, d.content = $content,
                    d.content_type = $content_type, d.timestamp = $timestamp
                WITH d
                MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                MERGE (m)-[:{schema.REL_HAS_DETAIL}]->(d)
            """, {
                "det_id": detail.detail_id,
                "role": detail.role,
                "content": detail.content,
                "content_type": detail.content_type,
                "timestamp": ts,
                "mem_id": memory_id,
            })

    def add_relationships(self, memory_id_1: str, memory_id_2: str,
                          relationship_type: str = "RELATES_TO", weight: float = 0.5):
        """Add a relationship between two Memory nodes."""
        rel = relationship_type if relationship_type in (
            schema.REL_RELATES_TO, schema.REL_SARAH_APPROVED
        ) else schema.REL_RELATES_TO
        with self.driver.session() as session:
            session.run(f"""
                MATCH (a:{schema.LABEL_MEMORY} {{memory_id: $mem_id_1}})
                MATCH (b:{schema.LABEL_MEMORY} {{memory_id: $mem_id_2}})
                MERGE (a)-[r:{rel}]->(b)
                SET r.weight = $weight
            """, {
                "mem_id_1": memory_id_1,
                "mem_id_2": memory_id_2,
                "weight": weight,
            })

    def update_relevance_score(self, memory_id: str, delta: float):
        """Update a memory's relevance score (for decay/revival)."""
        with self.driver.session() as session:
            session.run(f"""
                MATCH (m:{schema.LABEL_MEMORY} {{memory_id: $mem_id}})
                SET m.relevance_score = COALESCE(m.relevance_score, 1.0) + $delta
                SET m.relevance_score = MIN(MAX(m.relevance_score, 0.0), 2.0)
            """, {"mem_id": memory_id, "delta": delta})

    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()
