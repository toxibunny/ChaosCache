import pytest
from vibe_memory.retrieval.engine import RetrievalEngine, MemoryStore


class MockSession:
    """Mock Neo4j session for testing."""

    def __init__(self, mock_data=None):
        self.mock_data = mock_data or []
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def run(self, query, parameters=None):
        self.queries.append((query, parameters or {}))
        return MockResult(self.mock_data)


class MockResult:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for item in self.data:
            yield {"data": item}


class MockDriver:
    def __init__(self, mock_data=None):
        self.mock_data = mock_data

    def session(self):
        return MockSession(self.mock_data)


def test_retrieve_by_entities():
    mock_data = [
        {"memory_id": "mem-1", "text": "We went to the beach",
         "summary": "Beach day", "relevance_score": 1.0,
         "emotion_tags": ["happy"], "entities": ["beach"],
         "themes": [], "timestamp": 1700412034.0, "notable_quotes": []},
        {"memory_id": "mem-2", "text": "Saw crabs at the beach",
         "summary": "Crab sighting", "relevance_score": 0.8,
         "emotion_tags": ["excited"], "entities": ["crab", "beach"],
         "themes": [], "timestamp": 1700412035.0, "notable_quotes": []},
    ]
    driver = MockDriver(mock_data)
    engine = RetrievalEngine(driver)
    results = engine.retrieve(entities=["beach"], max_results=5)
    assert len(results) == 2
    assert results[0].memory_id == "mem-1"


def test_structured_query_by_emotion():
    mock_data = [
        {"memory_id": "mem-3", "text": "We laughed so hard",
         "summary": "Laughing fit", "relevance_score": 0.9,
         "emotion_tags": ["chaotic_giggles"], "entities": [],
         "themes": [], "timestamp": 1700412036.0, "notable_quotes": []},
    ]
    driver = MockDriver(mock_data)
    engine = RetrievalEngine(driver)
    results = engine.query(emotion_tags=["chaotic_giggles"], max_results=10)
    assert len(results) == 1
    assert "chaotic_giggles" in results[0].emotion_tags


def test_text_search():
    mock_data = [
        {"memory_id": "mem-10", "text": "Pickle the cat is awesome",
         "summary": "About Pickle", "relevance_score": 1.0,
         "emotion_tags": [], "entities": ["Pickle"],
         "themes": [], "timestamp": 1700412037.0, "notable_quotes": []},
        {"memory_id": "mem-11", "text": "My cat Pickle loves naps",
         "summary": "Pickle napping", "relevance_score": 0.9,
         "emotion_tags": [], "entities": ["Pickle"],
         "themes": [], "timestamp": 1700412038.0, "notable_quotes": []},
    ]
    driver = MockDriver(mock_data)
    engine = RetrievalEngine(driver)
    results = engine.search_text(term="Pickle", max_results=10)
    assert len(results) == 2
    assert all("Pickle" in r.text or "Pickle" in r.summary for r in results)


def test_serendipity_parameter():
    driver = MockDriver([])
    engine = RetrievalEngine(driver, serendipity=0.5)
    assert engine.serendipity == 0.5


def test_reminds_me_of():
    mock_data = [
        {"memory_id": "mem-20", "text": "That time with the crabs",
         "summary": "Crab memory", "relevance_score": 0.7,
         "emotion_tags": ["nostalgic"], "entities": ["crab"],
         "themes": [], "timestamp": 1700412039.0, "notable_quotes": []},
    ]
    driver = MockDriver(mock_data)
    engine = RetrievalEngine(driver)
    results = engine.reminds_me_of(current_topic="beach", max_results=3)
    # Even with mock, should not crash
    assert isinstance(results, list)
