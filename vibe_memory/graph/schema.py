"""Neo4j schema definitions for the Vibe Memory graph."""

# Node labels
LABEL_CHAPTER = "Chapter"
LABEL_CONVERSATION = "Conversation"
LABEL_MEMORY = "Memory"
LABEL_DETAIL = "Detail"
LABEL_ENTITY = "Entity"
LABEL_EMOTION = "Emotion"

# Relationship types
REL_CONTAINS = "CONTAINS"           # Chapter→Conversation, Conversation→Memory
REL_HAS_DETAIL = "HAS_DETAIL"       # Memory→Detail
REL_RELATES_TO = "RELATES_TO"       # Memory→Memory (cross-links)
REL_HAS_EMOTION = "HAS_EMOTION"     # Memory→Emotion
REL_MENTIONS = "MENTIONS"           # Memory→Entity
REL_APPEARS_WITH = "APPEARS_WITH"   # Entity→Entity (co-occurrence)
REL_SARAH_APPROVED = "SARAH_APPROVED"  # Memory→Memory (memories that slap)

# Schema creation Cypher
CREATE_CONSTRAINTS = """
CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE;
CREATE CONSTRAINT conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.conversation_id IS UNIQUE;
CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE;
CREATE CONSTRAINT detail_id IF NOT EXISTS FOR (d:Detail) REQUIRE d.detail_id IS UNIQUE;
CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;
CREATE CONSTRAINT emotion_name IF NOT EXISTS FOR (em:Emotion) REQUIRE em.name IS UNIQUE;

CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp);
CREATE INDEX memory_emotion_tags IF NOT EXISTS FOR (m:Memory) ON (m.emotion_tags);
CREATE INDEX memory_themes IF NOT EXISTS FOR (m:Memory) ON (m.themes);
CREATE INDEX memory_relevance IF NOT EXISTS FOR (m:Memory) ON (m.relevance_score);
CREATE INDEX entity_name_lookup IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX emotion_name_lookup IF NOT EXISTS FOR (em:Emotion) ON (em.name);
CREATE INDEX conversation_time IF NOT EXISTS FOR (c:Conversation) ON (c.create_time);
"""
