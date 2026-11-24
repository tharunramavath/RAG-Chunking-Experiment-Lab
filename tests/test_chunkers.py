"""
Tests for all chunking strategies.
"""

import pytest
from core.chunkers import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    SemanticChunker,
    ParentChildChunker,
    count_tokens,
)
from core.embeddings import FastTfidfEmbedder

SAMPLE_TEXT = """Artificial Intelligence and retrieval systems are transforming software engineering.
Vector search allows fast similarity computation across large knowledge bases.
Chunking is the critical step of breaking documents into searchable units.

Fixed size chunking slices text into equal segments regardless of syntax.
Recursive chunking uses hierarchical delimiters like double newlines and single newlines.
Sentence chunking keeps grammatical boundaries intact.
Semantic chunking calculates cosine distance between adjacent sentences.
Parent-child chunking indexes small children while retrieving large parent contexts.
"""


def test_token_counter():
    count = count_tokens("Hello world! This is a test.")
    assert count > 0
    assert count_tokens("") == 0


def test_fixed_size_chunker():
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20, unit="characters")
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 100
        assert c.token_count > 0

    # Overlap validation
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=50, chunk_overlap=50)


def test_fixed_size_token_unit():
    chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5, unit="tokens")
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 25


def test_recursive_chunker():
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200
        assert c.chunk_id.startswith("recursive_")


def test_sentence_chunker():
    chunker = SentenceChunker(sentences_per_chunk=2, sentence_overlap=1)
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["strategy"] == "sentence"
        assert len(c.text) > 0


def test_semantic_chunker():
    embedder = FastTfidfEmbedder()
    chunker = SemanticChunker(
        embedder=embedder,
        similarity_threshold=0.5,
        threshold_type="threshold",
        max_chunk_size=500,
    )
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata["strategy"] == "semantic"


def test_parent_child_chunker():
    chunker = ParentChildChunker(
        parent_chunk_size=300,
        parent_overlap=50,
        child_chunk_size=100,
        child_overlap=20,
    )
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 1
    for c in chunks:
        assert c.parent_id is not None
        assert c.parent_text is not None
        assert len(c.parent_text) >= len(c.text)
        assert c.context_text == c.parent_text
        assert c.retrieval_text == c.text
