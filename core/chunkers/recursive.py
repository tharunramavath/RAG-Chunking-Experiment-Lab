"""
Recursive Character Chunker: Splits text hierarchically using a prioritized list of separators.
Maintains natural paragraph, sentence, and word boundaries while respecting chunk_size and chunk_overlap.
"""

from typing import List, Optional
from core.chunkers.base import BaseChunker, Chunk, count_tokens


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        separators = separators or ["\n\n", "\n", ". ", " ", ""]
        super().__init__(
            "recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        # Find the first separator present in the text
        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        # Split on the chosen separator
        splits = text.split(separator) if separator != "" else list(text)

        # Merge splits up to chunk_size
        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    # Can't split further, hard slice
                    final_chunks.append(s[:self.chunk_size])
                else:
                    re_splits = self._split_text(s, new_separators)
                    final_chunks.extend(re_splits)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for s in splits:
            s_len = len(s) + (len(separator) if current_doc else 0)
            if total_len + s_len > self.chunk_size and current_doc:
                doc = separator.join(current_doc)
                if doc.strip():
                    docs.append(doc.strip())
                # Keep overlap from the end of current_doc
                while total_len > self.chunk_overlap and current_doc:
                    removed = current_doc.pop(0)
                    total_len -= len(removed) + (len(separator) if current_doc else 0)

            current_doc.append(s)
            total_len += len(s) + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc.strip())

        return docs

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        raw_chunks = self._split_text(text, self.separators)
        chunks: List[Chunk] = []
        for i, c in enumerate(raw_chunks):
            cleaned = c.strip()
            if cleaned:
                chunks.append(
                    Chunk(
                        chunk_id=f"recursive_{i}",
                        text=cleaned,
                        char_count=len(cleaned),
                        token_count=count_tokens(cleaned),
                        metadata={
                            "strategy": "recursive",
                            "chunk_index": i,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                        },
                    )
                )
        return chunks
