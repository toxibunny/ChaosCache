"""Relevance scoring for memory retrieval."""

import math
from datetime import datetime, timezone
from typing import Optional

from vibe_memory.models import Memory


class RelevanceScorer:
    """Scores memories based on query context.

    Combines entity overlap, emotion overlap, text similarity, and recency
    into a single relevance score between 0.0 and 1.0.
    """

    def __init__(
        self,
        entity_weight: float = 0.4,
        emotion_weight: float = 0.2,
        text_weight: float = 0.3,
        recency_weight: float = 0.1,
        recency_half_life_days: float = 30.0,
    ):
        """
        Args:
            entity_weight: Weight for entity overlap (0-1)
            emotion_weight: Weight for emotion overlap (0-1)
            text_weight: Weight for text similarity (0-1)
            recency_weight: Weight for recency boost (0-1)
            recency_half_life_days: Days until recency factor drops to 0.5
        """
        total = entity_weight + emotion_weight + text_weight + recency_weight
        self.entity_weight = entity_weight / total
        self.emotion_weight = emotion_weight / total
        self.text_weight = text_weight / total
        self.recency_weight = recency_weight / total
        self.recency_half_life = recency_half_life_days

    def score(
        self,
        memory: Memory,
        query_entities: Optional[list[str]] = None,
        query_emotions: Optional[list[str]] = None,
        query_text: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> float:
        """Score a memory against the query context.

        Returns a score between 0.0 and 1.0.
        """
        query_entities = query_entities or []
        query_emotions = query_emotions or []
        query_text = query_text or ""
        now = now or datetime.now(timezone.utc)

        # Entity overlap score (Jaccard similarity)
        entity_score = self._jaccard(query_entities, memory.entities)

        # Emotion overlap score
        emotion_score = self._jaccard(query_emotions, memory.emotion_tags)

        # Text similarity score (word overlap with TF-IDF-like weighting)
        text_score = self._text_similarity(query_text, memory.text, memory.summary)

        # Recency score (exponential decay)
        recency_score = self._recency_score(memory.timestamp, now)

        # Weighted combination
        score = (
            self.entity_weight * entity_score
            + self.emotion_weight * emotion_score
            + self.text_weight * text_score
            + self.recency_weight * recency_score
        )

        # Apply boost as a multiplier (stored importance/frequency modifier)
        score *= memory.boost

        return max(0.0, min(2.0, score))  # Allow up to 2.0 with boost

    def _jaccard(self, set1: list[str], set2: list[str]) -> float:
        """Jaccard similarity between two sets of strings."""
        if not set1 or not set2:
            return 0.0

        s1 = {s.lower().strip() for s in set1 if s.strip()}
        s2 = {s.lower().strip() for s in set2 if s.strip()}

        if not s1 or not s2:
            return 0.0

        intersection = len(s1 & s2)
        union = len(s1 | s2)

        return intersection / union if union > 0 else 0.0

    def _text_similarity(self, query: str, text: str, summary: str) -> float:
        """Simple word overlap similarity with IDF-like weighting."""
        if not query.strip():
            return 0.0

        query_words = self._tokenize(query)
        if not query_words:
            return 0.0

        # Use summary if available (more concise), otherwise use full text
        target = summary if summary else text
        target_words = self._tokenize(target)
        if not target_words:
            return 0.0

        # Count overlaps
        query_set = set(query_words)
        target_set = set(target_words)

        # Simple overlap ratio
        overlap = len(query_set & target_set)
        max_possible = min(len(query_set), len(target_set))

        return overlap / max_possible if max_possible > 0 else 0.0

    def _recency_score(self, timestamp: Optional[datetime], now: datetime) -> float:
        """Exponential decay based on age."""
        if not timestamp:
            return 0.5  # Neutral score for unknown timestamps

        # Handle naive vs aware datetime
        if timestamp.tzinfo is None and now.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        age_days = (now - timestamp).total_seconds() / 86400
        if age_days < 0:
            age_days = 0

        # Exponential decay: score = 2^(-age/half_life)
        return 2 ** (-age_days / self.recency_half_life)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        import re
        return [w.lower() for w in re.findall(r'\b[a-z0-9]+\b', text.lower()) if len(w) > 1]

    def rank(self, memories: list[Memory], **kwargs) -> list[Memory]:
        """Rank memories by relevance score (descending).

        Modifies the boost attribute of each memory and returns
        the sorted list.
        """
        scored = []
        for mem in memories:
            score = self.score(mem, **kwargs)
            mem.boost = score
            scored.append((score, mem))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored]
