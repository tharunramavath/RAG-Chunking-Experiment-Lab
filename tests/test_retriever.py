"""
Tests for VectorIndex and retrieval logic.
"""

from core.chunkers import RecursiveChunker, ParentChildChunker
from core.embeddings import FastTfidfEmbedder
from core.retriever import VectorIndex

SAMPLE_DOC = """
Section 1: The speed of light in vacuum is approximately 299,792,458 meters per second.
This is a universal physical constant denoted by the letter c.

Section 2: Photosynthesis is the biological process used by plants and other organisms
to convert light energy into chemical energy stored in glucose.
"""


def test_vector_index_search():
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk(SAMPLE_DOC)

    embedder = FastTfidfEmbedder()
    index = VectorIndex(chunks, embedder)

    assert index.indexing_time_ms >= 0.0

    output = index.query("speed of light in vacuum", top_k=2)
    assert len(output.top_k_results) > 0
    assert output.retrieval_time_ms >= 0.0
    assert "299,792,458" in output.top_k_results[0].context_text


def test_parent_child_retrieval_resolution():
    chunker = ParentChildChunker(
        parent_chunk_size=300,
        parent_overlap=50,
        child_chunk_size=100,
        child_overlap=20,
    )
    chunks = chunker.chunk(SAMPLE_DOC)

    embedder = FastTfidfEmbedder()
    index = VectorIndex(chunks, embedder)

    output = index.query("photosynthesis biological process", top_k=1, resolve_parents=True)
    assert len(output.top_k_results) == 1
    top_res = output.top_k_results[0]
    assert top_res.is_parent_retrieval is True
    # The returned context_text must be the parent text (richer context)
    assert "chemical energy stored in glucose" in top_res.context_text
