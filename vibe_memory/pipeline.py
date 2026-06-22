"""Pipeline — orchestrates the full ingestion pipeline."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from vibe_memory.ingestor import Ingestor
from vibe_memory.summarizer.base import Summarizer
from vibe_memory.summarizer.llama_cpp import LlamaCppSummarizer
from vibe_memory.graph.builder import GraphBuilder
from vibe_memory.models import Memory, Conversation, Chapter, Detail

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    db_path: str
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    model_path: str = ""
    chunk_size: int = 10
    summarizer_backend: str = "llama_cpp"  # llama_cpp, ollama, openai
    dry_run: bool = False
    chapter_grouping: str = "monthly"  # monthly, yearly, manual


async def run_ingestion(config: IngestionConfig, summarizer: Optional[Summarizer] = None) -> dict:
    """Run the full ingestion pipeline.

    Pipeline: Ingestor → Summarizer → GraphBuilder → Neo4j

    Args:
        config: Ingestion configuration
        summarizer: Optional pre-configured summarizer (for testing)

    Returns:
        Stats dict with counts and error information
    """
    stats = {
        "conversations_processed": 0,
        "memories_created": 0,
        "chapters_created": 0,
        "errors": 0,
    }

    if config.dry_run:
        logger.info("Dry run mode — no Neo4j or LLM calls")
        return stats

    # Open connections
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(config.neo4j_url, auth=(config.neo4j_user, config.neo4j_password))
    builder = GraphBuilder(driver)

    try:
        # Create schema
        builder.create_schema()

        # Initialize summarizer
        if summarizer is None:
            summarizer = LlamaCppSummarizer(model_path=config.model_path)

        # Open ingestor
        with Ingestor(config.db_path, chunk_size=config.chunk_size) as ingestor:
            conversations = ingestor.load_conversations()
            logger.info(f"Loaded {len(conversations)} conversations")

            # Group conversations into chapters
            chapters = _group_into_chapters(conversations, config.chapter_grouping)

            for chapter in chapters:
                # Summarize each conversation in the chapter
                conv_summaries = []
                for conv in chapter.conversations:
                    chunks = ingestor.chunk_conversation(conv)
                    chunks_text = [ingestor._messages_to_text(c) for c in chunks]

                    try:
                        conv_summary = await summarizer.summarize_conversation(chunks_text)
                        conv_summaries.append(conv_summary)
                    except Exception as e:
                        logger.warning(f"Failed to summarize conversation {conv.conversation_id}: {e}")
                        stats["errors"] += 1
                        conv_summaries.append(
                            type(conv_summary)(title=conv.title, summary=conv.title)
                        )

                # Build chapter summary
                try:
                    chapter_summary = await summarizer.summarize_chapter(conv_summaries)
                except Exception as e:
                    logger.warning(f"Failed to summarize chapter: {e}")
                    stats["errors"] += 1
                    chapter_summary = type(chapter_summary)(title=chapter.title)

                # Add chapter to graph
                print(f"\n📖 Chapter: {chapter.title} ({len(chapter.conversations)} conversations)")
                chapter_node = Chapter(
                    chapter_id=chapter.chapter_id,
                    title=chapter_summary.title or chapter.title,
                    start_time=chapter.start_time,
                    end_time=chapter.end_time,
                    summary=chapter_summary.summary,
                    themes=chapter_summary.recurring_themes,
                    emotion_tags=[],
                )
                builder.add_chapter(chapter_node)
                stats["chapters_created"] += 1

                # Process each conversation's chunks as memories
                for conv_idx, conv in enumerate(chapter.conversations):
                    # Add conversation node (only once per conversation)
                    if conv_idx == 0 or conv.conversation_id != chapter.conversations[conv_idx-1].conversation_id:
                        conv_node = Conversation(
                            conversation_id=conv.conversation_id,
                            title=conv.title,
                            create_time=conv.create_time,
                            model_slug=conv.model_slug,
                            message_count=conv.message_count,
                            summary=conv.summary,
                            is_archived=conv.is_archived,
                        )
                        builder.add_conversation(conv_node, chapter.chapter_id)
                        stats["conversations_processed"] += 1
                        print(f"  💬 {conv.title}")

                    chunks = ingestor.chunk_conversation(conv)
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_text = ingestor._messages_to_text(chunk)

                        # Extract memory
                        try:
                            extraction = await summarizer.extract_memory(chunk_text)
                        except Exception as e:
                            logger.warning(f"Failed to extract memory: {e}")
                            stats["errors"] += 1
                            extraction = type(extraction)(summary=chunk_text[:200])

                        # Create memory node
                        memory = Memory(
                            memory_id=str(uuid.uuid4()),
                            text=chunk_text,
                            summary=extraction.summary,
                            entities=extraction.entities,
                            emotion_tags=extraction.emotion_tags,
                            themes=extraction.themes,
                            timestamp=datetime.now(timezone.utc),
                            relevance_score=1.0,
                            notable_quotes=extraction.notable_quotes,
                        )
                        builder.add_memory(memory, conv.conversation_id)
                        stats["memories_created"] += 1

                        # Display memory with context
                        conv_date = datetime.fromtimestamp(conv.create_time, tz=timezone.utc).strftime('%Y-%m-%d')
                        summary = extraction.summary[:80].replace('\n', ' ')
                        emotions = f" [{', '.join(extraction.emotion_tags)}]" if extraction.emotion_tags else ""
                        print(f"  🧠 [{conv_date}] {conv.title[:40]:<40} | {summary}{emotions}")

        logger.info(f"Ingestion complete: {stats}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        builder.close()

    return stats


def _group_into_chapters(conversations: list[Conversation], grouping: str):
    """Group conversations into chapters by time period."""

    @dataclass
    class ChapterGroup:
        chapter_id: str
        title: str
        start_time: float
        end_time: float
        conversations: list[Conversation] = field(default_factory=list)
        period_key: str = ""

    if not conversations:
        return []

    chapters = []
    current_chapter = None

    for conv in conversations:
        if grouping == "monthly":
            dt = datetime.fromtimestamp(conv.create_time, tz=timezone.utc)
            period_key = dt.strftime("%Y-%m")
        elif grouping == "yearly":
            dt = datetime.fromtimestamp(conv.create_time, tz=timezone.utc)
            period_key = dt.strftime("%Y")
        else:
            period_key = "all"

        if current_chapter is None or current_chapter.period_key != period_key:
            if current_chapter:
                chapters.append(current_chapter)
            dt = datetime.fromtimestamp(conv.create_time, tz=timezone.utc)
            current_chapter = ChapterGroup(
                chapter_id=str(uuid.uuid4()),
                title=period_key,
                start_time=conv.create_time,
                end_time=conv.create_time,
                period_key=period_key,
            )

        current_chapter.conversations.append(conv)
        current_chapter.end_time = max(current_chapter.end_time, conv.create_time)

    if current_chapter:
        chapters.append(current_chapter)

    return chapters
