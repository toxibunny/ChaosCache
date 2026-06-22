import pytest
from vibe_memory.pipeline import IngestionConfig, run_ingestion


def test_ingestion_config():
    config = IngestionConfig(
        db_path="test.db",
        neo4j_url="bolt://localhost:7687",
        model_path="/path/to/model.gguf",
        chunk_size=10,
    )
    assert config.chunk_size == 10
    assert config.db_path == "test.db"


def test_run_ingestion_with_mock():
    """Test the pipeline orchestrator with mock components."""
    config = IngestionConfig(
        db_path=":memory:",  # in-memory DB
        neo4j_url="bolt://localhost:7687",
        model_path="/dummy.gguf",
        chunk_size=5,
        dry_run=True,
    )
    # Should not crash even with empty DB and no model
    import asyncio
    result = asyncio.run(run_ingestion(config))
    assert result["conversations_processed"] >= 0
    assert result["errors"] >= 0
