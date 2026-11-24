"""
Base classes and interfaces for chunking strategies in the RAG Chunking Lab.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
# Reusable tiktoken encoder with safe fallback if tiktoken is not installed
try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODER = None


def count_tokens(text: str) -> int:
    """Accurately count tokens using cl100k_base (OpenAI standard) with heuristic fallback."""
    if not text:
        return 0
    if _TIKTOKEN_ENCODER is not None:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text))
        except Exception:
            pass
    # Conservative heuristic fallback: ~4 characters per token
    return max(1, len(text.split()))


@dataclass
class Chunk:
    """Represents a discrete text chunk produced by any chunking strategy."""
    chunk_id: str
    text: str
    char_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.token_count:
            self.token_count = count_tokens(self.text)

    @property
    def retrieval_text(self) -> str:
        """Text used for vector indexing / matching."""
        return self.text

    @property
    def context_text(self) -> str:
        """Text provided to the generator / returned as RAG context."""
        return self.parent_text if self.parent_text else self.text


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split input document text into a list of Chunk objects."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return strategy configuration parameters."""
        return {"strategy": self.name, **self.params}
