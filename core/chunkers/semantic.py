"""
Semantic Chunker: Detects semantic topic shifts using embedding distances between consecutive sentences.
Splits occur at valleys of cosine similarity (or peaks of semantic distance).
"""

import numpy as np
from typing import List, Optional, Any
from core.chunkers.base import BaseChunker, Chunk, count_tokens
from core.chunkers.sentence import split_into_sentences


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        embedder: Optional[Any] = None,
        similarity_threshold: float = 0.65,
        threshold_type: str = "percentile",  # "threshold" or "percentile"
        threshold_percentile: float = 75.0,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
    ):
        super().__init__(
            "semantic",
            similarity_threshold=similarity_threshold,
            threshold_type=threshold_type,
            threshold_percentile=threshold_percentile,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.threshold_type = threshold_type
        self.threshold_percentile = threshold_percentile
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def set_embedder(self, embedder: Any):
        self.embedder = embedder

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        sentences = split_into_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                Chunk(
                    chunk_id="semantic_0",
                    text=sentences[0],
                    char_count=len(sentences[0]),
                    token_count=count_tokens(sentences[0]),
                    metadata={"strategy": "semantic", "chunk_index": 0},
                )
            ]

        # Compute embeddings for each sentence
        if self.embedder is None:
            # Fallback to simple TF-IDF / character n-gram embeddings if no embedder provided
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            embeddings = vectorizer.fit_transform(sentences).toarray()
        else:
            embeddings = self.embedder.encode(sentences)
            if hasattr(embeddings, "cpu"):
                embeddings = embeddings.cpu().numpy()
            embeddings = np.array(embeddings)

        # Calculate cosine similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Determine cutoff threshold
        if self.threshold_type == "percentile" and len(similarities) > 1:
            # Splits occur where similarity is in the lower percentiles (e.g. bottom 25% similarity = high distance)
            cutoff = float(np.percentile(similarities, 100 - self.threshold_percentile))
        else:
            cutoff = self.similarity_threshold

        chunks: List[Chunk] = []
        current_sentences = [sentences[0]]
        current_len = len(sentences[0])
        chunk_num = 0

        for i in range(len(similarities)):
            sim = similarities[i]
            next_sent = sentences[i + 1]
            next_len = len(next_sent)

            # Split conditions:
            # 1. Similarity drops below cutoff and current chunk is at least min_chunk_size
            # 2. Or current chunk exceeds max_chunk_size
            should_split = False
            if sim < cutoff and current_len >= self.min_chunk_size:
                should_split = True
            elif current_len + next_len > self.max_chunk_size:
                should_split = True

            if should_split:
                chunk_str = " ".join(current_sentences).strip()
                if chunk_str:
                    chunks.append(
                        Chunk(
                            chunk_id=f"semantic_{chunk_num}",
                            text=chunk_str,
                            char_count=len(chunk_str),
                            token_count=count_tokens(chunk_str),
                            metadata={
                                "strategy": "semantic",
                                "similarity_cutoff": cutoff,
                                "chunk_index": chunk_num,
                                "sentence_count": len(current_sentences),
                            },
                        )
                    )
                    chunk_num += 1
                current_sentences = [next_sent]
                current_len = next_len
            else:
                current_sentences.append(next_sent)
                current_len += next_len + 1

        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            if chunk_str:
                chunks.append(
                    Chunk(
                        chunk_id=f"semantic_{chunk_num}",
                        text=chunk_str,
                        char_count=len(chunk_str),
                        token_count=count_tokens(chunk_str),
                        metadata={
                            "strategy": "semantic",
                            "similarity_cutoff": cutoff,
                            "chunk_index": chunk_num,
                            "sentence_count": len(current_sentences),
                        },
                    )
                )

        return chunks
