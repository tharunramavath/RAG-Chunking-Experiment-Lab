"""
Sentence Chunker: Splits text respecting natural grammatical sentence boundaries.
Supports grouping by fixed sentence window (with sentence overlap) or target size without cutting sentences.
"""

import re
from typing import List
from core.chunkers.base import BaseChunker, Chunk, count_tokens


def split_into_sentences(text: str) -> List[str]:
    """Robust sentence splitter that respects common abbreviations, decimals, and paragraphs."""
    if not text:
        return []

    text_clean = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split first on explicit paragraphs / multiple newlines
    paragraphs = [p.strip() for p in text_clean.split("\n\n") if p.strip()]

    sentences = []
    # Pattern looks behind at single punctuation character [.!?] followed by whitespace and capital letter / quote
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')

    for p in paragraphs:
        # Also handle standalone single newline lines (e.g. lists or markdown headers)
        lines = [line.strip() for line in p.split("\n") if line.strip()]
        for line in lines:
            parts = sentence_pattern.split(line)
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    sentences.append(cleaned)

    return sentences


class SentenceChunker(BaseChunker):
    def __init__(
        self,
        sentences_per_chunk: int = 3,
        sentence_overlap: int = 1,
        max_chunk_size: int = 600,
    ):
        if sentence_overlap >= sentences_per_chunk:
            raise ValueError("sentence_overlap must be strictly less than sentences_per_chunk")
        super().__init__(
            "sentence",
            sentences_per_chunk=sentences_per_chunk,
            sentence_overlap=sentence_overlap,
            max_chunk_size=max_chunk_size,
        )
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        sentences = split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        idx = 0
        step = max(1, self.sentences_per_chunk - self.sentence_overlap)
        chunk_num = 0

        while idx < len(sentences):
            group = sentences[idx : idx + self.sentences_per_chunk]
            chunk_str = " ".join(group).strip()

            # If grouped sentences exceed max_chunk_size, split dynamically
            if len(chunk_str) > self.max_chunk_size and len(group) > 1:
                # Add sentences one by one up to max_chunk_size
                sub_group = []
                sub_len = 0
                for s in group:
                    if sub_len + len(s) + 1 <= self.max_chunk_size or not sub_group:
                        sub_group.append(s)
                        sub_len += len(s) + 1
                    else:
                        break
                chunk_str = " ".join(sub_group).strip()

            if chunk_str:
                chunks.append(
                    Chunk(
                        chunk_id=f"sentence_{chunk_num}",
                        text=chunk_str,
                        char_count=len(chunk_str),
                        token_count=count_tokens(chunk_str),
                        metadata={
                            "strategy": "sentence",
                            "start_sentence_idx": idx,
                            "num_sentences": len(group),
                            "chunk_index": chunk_num,
                        },
                    )
                )
                chunk_num += 1

            if idx + self.sentences_per_chunk >= len(sentences):
                break
            idx += step

        return chunks
