"""
Parent-Child (Hierarchical) Chunker:
Splits text into larger Parent chunks (for rich context) and smaller Child chunks (for high-resolution vector search).
Child chunks carry a pointer to their Parent chunk.
"""

from typing import List
from core.chunkers.base import BaseChunker, Chunk, count_tokens
from core.chunkers.recursive import RecursiveChunker


class ParentChildChunker(BaseChunker):
    def __init__(
        self,
        parent_chunk_size: int = 1200,
        parent_overlap: int = 100,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ):
        if parent_overlap >= parent_chunk_size:
            raise ValueError("parent_overlap must be strictly less than parent_chunk_size")
        if child_overlap >= child_chunk_size:
            raise ValueError("child_overlap must be strictly less than child_chunk_size")
        if child_chunk_size > parent_chunk_size:
            raise ValueError("child_chunk_size cannot exceed parent_chunk_size")

        super().__init__(
            "parent_child",
            parent_chunk_size=parent_chunk_size,
            parent_overlap=parent_overlap,
            child_chunk_size=child_chunk_size,
            child_overlap=child_overlap,
        )
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

        self._parent_splitter = RecursiveChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
        )
        self._child_splitter = RecursiveChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
        )

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        parent_chunks = self._parent_splitter.chunk(text)
        all_child_chunks: List[Chunk] = []

        for p_idx, parent in enumerate(parent_chunks):
            parent_id = f"parent_{p_idx}"
            children = self._child_splitter.chunk(parent.text)

            for c_idx, child in enumerate(children):
                child_id = f"{parent_id}_child_{c_idx}"
                all_child_chunks.append(
                    Chunk(
                        chunk_id=child_id,
                        text=child.text,
                        char_count=len(child.text),
                        token_count=count_tokens(child.text),
                        parent_id=parent_id,
                        parent_text=parent.text,
                        metadata={
                            "strategy": "parent_child",
                            "parent_id": parent_id,
                            "parent_index": p_idx,
                            "child_index": c_idx,
                            "parent_tokens": parent.token_count,
                            "child_tokens": child.token_count,
                        },
                    )
                )

        return all_child_chunks
