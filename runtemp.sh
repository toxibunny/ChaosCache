#!/bin/bash
# Run the ChaosCache ingestion pipeline
cd "$(dirname "$0")"

python3 -c "
import asyncio
from vibe_memory import IngestionConfig, run_ingestion
from vibe_memory.summarizer.llama_server import LlamaServerSummarizer

config = IngestionConfig(
    db_path='chats.db',
    neo4j_url='bolt://localhost:7687',
    neo4j_user='neo4j',
    neo4j_password='password',
    model_path='',
    chunk_size=10,
    chapter_grouping='monthly',
)

summarizer = LlamaServerSummarizer(
    server_url='http://localhost:8081',
    temperature=0.3,
)

async def main():
    stats = await run_ingestion(config, summarizer=summarizer)
    print(f'Done: {stats}')

asyncio.run(main())
"
