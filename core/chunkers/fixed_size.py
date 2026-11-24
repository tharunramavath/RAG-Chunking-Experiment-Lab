"""
Fixed-size chunker: splits text into chunks of exact size with specified overlap.
Supports both character-based and token-based units.
"""

from typing import List
from core.chunkers.base import BaseChunker, Chunk, count_tokens, _TIKTOKEN_ENCODER


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, unit: str = "characters"):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        super().__init__("fixed_size", chunk_size=chunk_size, chunk_overlap=chunk_overlap, unit=unit)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit.lower()

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap

        if self.unit == "tokens" and _TIKTOKEN_ENCODER is not None:
            tokens = _TIKTOKEN_ENCODER.encode(text)
            total_tokens = len(tokens)
            idx = 0
            chunk_num = 0

            while idx < total_tokens:
                end_idx = min(idx + self.chunk_size, total_tokens)
                chunk_token_ids = tokens[idx:end_idx]
                chunk_str = _TIKTOKEN_ENCODER.decode(chunk_token_ids).strip()
                if chunk_str:
                    chunks.append(
                        Chunk(
                            chunk_id=f"fixed_{chunk_num}",
                            text=chunk_str,
                            token_count=len(chunk_token_ids),
                            char_count=len(chunk_str),
                            metadata={
                                "strategy": "fixed_size",
                                "unit": "tokens",
                                "start_token": idx,
                                "end_token": end_idx,
                                "chunk_index": chunk_num,
                            },
                        )
                    )
                    chunk_num += 1
                if end_idx >= total_tokens:
                    break
                idx += step
        else:
            # Character-based fixed size
            total_chars = len(text)
            idx = 0
            chunk_num = 0

            while idx < total_chars:
                end_idx = min(idx + self.chunk_size, total_chars)
                chunk_str = text[idx:end_idx].strip()
                if chunk_str:
                    chunks.append(
                        Chunk(
                            chunk_id=f"fixed_{chunk_num}",
                            text=chunk_str,
                            token_count=count_tokens(chunk_str),
                            char_count=len(chunk_str),
                            metadata={
                                "strategy": "fixed_size",
                                "unit": "characters",
                                "start_char": idx,
                                "end_char": end_idx,
                                "chunk_index": chunk_num,
                            },
                        )
                    )
                    chunk_num += 1
                if end_idx >= total_chars:
                    break
                idx += step

        return chunks
