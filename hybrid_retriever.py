import time
import numpy as np
from langchain_community.retrievers import BM25Retriever

from config import BM25_WEIGHT, DENSE_WEIGHT, TOP_K


class HybridRetriever:
    """
    Manual BM25 + FAISS ensemble — no EnsembleRetriever dependency,
    works across all langchain versions.

    BM25  = sparse / keyword match  -> good for exact financial terms
    FAISS = dense  / semantic match -> good for paraphrased queries

    Merges by weighted Reciprocal Rank Fusion:
      score = bm25_weight*(1/rank_bm25) + dense_weight*(1/rank_dense)
    Deduplicates by page_content before returning top-k.
    """

    def __init__(self, docs, vectorstore, bm25_weight=BM25_WEIGHT,
                 dense_weight=DENSE_WEIGHT, k=TOP_K):
        self._bm25       = BM25Retriever.from_documents(docs)
        self._bm25.k     = k
        self._dense      = vectorstore.as_retriever(search_kwargs={"k": k})
        self._bw         = bm25_weight
        self._dw         = dense_weight
        self._k          = k
        self._latencies_ms = []

    # ------------------------------------------------------------------ Core
    def retrieve(self, query: str) -> list:
        """Retrieve and merge docs, recording latency."""
        start      = time.perf_counter()
        bm25_docs  = self._bm25.invoke(query)
        dense_docs = self._dense.invoke(query)

        # Weighted RRF merge -- deduplicate by content
        scores = {}
        seen   = {}
        for rank, doc in enumerate(bm25_docs, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + self._bw * (1 / rank)
            seen[key]   = doc
        for rank, doc in enumerate(dense_docs, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + self._dw * (1 / rank)
            seen[key]   = doc

        merged = sorted(seen.keys(), key=lambda k: scores[k], reverse=True)
        result = [seen[k] for k in merged[: self._k]]

        self._latencies_ms.append((time.perf_counter() - start) * 1_000)
        return result

    # ------------------------------------------------------------------ Stats
    @property
    def avg_latency_ms(self) -> float:
        """Mean retrieval latency across all calls. Use this for your resume."""
        if not self._latencies_ms:
            return 0.0
        return round(float(np.mean(self._latencies_ms)), 1)

    @property
    def p99_latency_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        return round(float(np.percentile(self._latencies_ms, 99)), 1)

    @property
    def stats(self) -> dict:
        return {
            "calls": len(self._latencies_ms),
            "avg_ms": self.avg_latency_ms,
            "p99_ms": self.p99_latency_ms,
        }
