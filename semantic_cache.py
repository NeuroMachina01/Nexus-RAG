import json
import os
import numpy as np
from typing import Optional


class SemanticCache:
    """
    Stores (query_embedding, answer, sources) tuples on disk.
    On lookup, computes cosine similarity between the incoming query
    and all cached queries. If similarity >= threshold, returns the
    cached answer — no LLM call needed.

    Tracks total queries and hits so you can measure cache_hit_rate
    and paste the real number into your resume.
    """

    def __init__(self, embeddings, threshold: float = 0.85, cache_file: str = "semantic_cache.json"):
        self.embeddings  = embeddings
        self.threshold   = threshold
        self.cache_file  = cache_file
        self.entries: list[dict] = []   # {query, embedding, answer, sources}
        self.total_queries = 0
        self.cache_hits    = 0
        self._load()

    # ------------------------------------------------------------------ I/O
    def _load(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.entries = json.load(f)

    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.entries, f)

    # ------------------------------------------------------------------ Core
    def lookup(self, query: str) -> Optional[dict]:
        """Return cached entry if a similar query exists, else None."""
        self.total_queries += 1
        if not self.entries:
            return None

        query_emb   = np.array(self.embeddings.embed_query(query))
        cached_embs = np.array([e["embedding"] for e in self.entries])

        # Cosine similarity (robust to zero vectors)
        norms  = np.linalg.norm(cached_embs, axis=1) * np.linalg.norm(query_emb)
        norms  = np.where(norms == 0, 1e-8, norms)
        scores = cached_embs.dot(query_emb) / norms

        best = int(np.argmax(scores))
        if scores[best] >= self.threshold:
            self.cache_hits += 1
            return self.entries[best]
        return None

    def store(self, query: str, answer: str, sources: str):
        """Persist a new query → answer pair."""
        embedding = self.embeddings.embed_query(query)
        self.entries.append({
            "query":     query,
            "embedding": embedding,
            "answer":    answer,
            "sources":   sources,
        })
        self._save()

    # ------------------------------------------------------------------ Stats
    @property
    def hit_rate(self) -> float:
        """Fraction of queries served from cache. Use this for your resume."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def stats(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "cache_hits":    self.cache_hits,
            "hit_rate_pct":  round(self.hit_rate * 100, 1),
            "cached_entries": len(self.entries),
        }
