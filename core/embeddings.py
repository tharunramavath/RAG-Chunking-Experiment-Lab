"""
Unified Embeddings Provider module.
Enforces an identical embedding model and dimension across all chunking strategies
for fair, controlled experiments.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseEmbedder(ABC):
    """Abstract embedding interface."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into a 2D float32 numpy array [N, D]."""
        pass

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a 1D float32 numpy array [D]."""
        vecs = self.encode([query])
        return vecs[0]


class FastTfidfEmbedder(BaseEmbedder):
    """
    Lightweight, deterministic TF-IDF dense embedder.
    Requires no external downloads, runs in milliseconds,
    and enables 100% reproducible offline benchmarking and fast tests.
    """

    def __init__(self, corpus: Optional[List[str]] = None):
        super().__init__(model_name="tfidf-dense-local")
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            min_df=1,
            max_features=512,
        )
        self.is_fitted = False
        if corpus:
            self.fit(corpus)

    def fit(self, corpus: List[str]):
        if corpus:
            self.vectorizer.fit(corpus)
            self.is_fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        if not self.is_fitted:
            # Self-fit on the provided texts if not fitted yet
            self.vectorizer.fit(texts)
            self.is_fitted = True
        sparse = self.vectorizer.transform(texts)
        dense = sparse.toarray().astype(np.float32)
        # Ensure L2 normalization
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Dense neural embedder backed by SentenceTransformers (e.g. all-MiniLM-L6-v2).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name=model_name)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


_EMBEDDER_CACHE = {}


def get_embedder(model_choice: str = "all-MiniLM-L6-v2", corpus: Optional[List[str]] = None) -> BaseEmbedder:
    """Factory function returning cached embedders."""
    key = model_choice.lower()
    if key in ("tfidf", "tfidf-dense-local", "fast-local"):
        if "tfidf" not in _EMBEDDER_CACHE or corpus:
            _EMBEDDER_CACHE["tfidf"] = FastTfidfEmbedder(corpus=corpus)
        return _EMBEDDER_CACHE["tfidf"]
    else:
        if key not in _EMBEDDER_CACHE:
            try:
                _EMBEDDER_CACHE[key] = SentenceTransformerEmbedder(model_name=model_choice)
            except Exception as e:
                # If network or model loading fails, gracefully fallback
                print(f"Warning: Failed to load {model_choice} ({e}). Falling back to FastTfidfEmbedder.")
                _EMBEDDER_CACHE[key] = FastTfidfEmbedder(corpus=corpus)
        return _EMBEDDER_CACHE[key]
