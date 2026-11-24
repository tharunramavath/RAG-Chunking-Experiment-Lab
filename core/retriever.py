"""
Vector Index and Retriever for RAG experiments.
Performs exact cosine similarity search over chunk embeddings and manages parent-child resolution.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from core.chunkers.base import Chunk, count_tokens
from core.embeddings import BaseEmbedder


@dataclass
class RetrievalResult:
    """Represents a single retrieved item in Top-K."""
    rank: int
    score: float
    chunk: Chunk
    context_text: str
    context_tokens: int
    is_parent_retrieval: bool = False


@dataclass
class QueryRetrievalOutput:
    """Represents full retrieval results for a single query."""
    query: str
    top_k_results: List[RetrievalResult]
    total_context_tokens: int
    retrieval_time_ms: float


class VectorIndex:
    def __init__(self, chunks: List[Chunk], embedder: BaseEmbedder):
        self.chunks = chunks
        self.embedder = embedder
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.indexing_time_ms: float = 0.0

        if chunks:
            self._build_index()

    def _build_index(self):
        t0 = time.perf_counter()
        texts_to_embed = [c.retrieval_text for c in self.chunks]
        self.embeddings = self.embedder.encode(texts_to_embed)
        self.indexing_time_ms = (time.perf_counter() - t0) * 1000.0

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        resolve_parents: bool = True,
        deduplicate_parents: bool = True,
    ) -> QueryRetrievalOutput:
        """
        Retrieve top_k chunks matching query_text using exact cosine similarity.
        Accurately measures retrieval time in milliseconds.
        """
        if not self.chunks or len(self.embeddings) == 0:
            return QueryRetrievalOutput(
                query=query_text,
                top_k_results=[],
                total_context_tokens=0,
                retrieval_time_ms=0.0,
            )

        t0 = time.perf_counter()
        q_vec = self.embedder.encode_query(query_text)

        # Dot product of normalized vectors = Cosine similarity
        scores = np.dot(self.embeddings, q_vec)

        # Get top-k indices
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1]

        results: List[RetrievalResult] = []
        seen_parents = set()
        rank = 1

        for idx in top_indices:
            if rank > k:
                break
            score = float(scores[idx])
            chunk = self.chunks[idx]

            is_parent = False
            if resolve_parents and chunk.parent_id and chunk.parent_text:
                if deduplicate_parents and chunk.parent_id in seen_parents:
                    # Skip duplicate parent in top-k
                    continue
                seen_parents.add(chunk.parent_id)
                ctx = chunk.parent_text
                is_parent = True
            else:
                ctx = chunk.text

            ctx_tokens = count_tokens(ctx)
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=score,
                    chunk=chunk,
                    context_text=ctx,
                    context_tokens=ctx_tokens,
                    is_parent_retrieval=is_parent,
                )
            )
            rank += 1

        t_end = time.perf_counter()
        retrieval_time_ms = (t_end - t0) * 1000.0
        total_tokens = sum(r.context_tokens for r in results)

        return QueryRetrievalOutput(
            query=query_text,
            top_k_results=results,
            total_context_tokens=total_tokens,
            retrieval_time_ms=retrieval_time_ms,
        )
