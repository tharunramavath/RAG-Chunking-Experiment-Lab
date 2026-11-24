"""
End-to-end integration test with demo dataset and all 5 chunking strategies.
"""

from core.dataset import load_demo_document, load_evaluation_questions
from core.embeddings import FastTfidfEmbedder
from core.retriever import VectorIndex
from core.evaluation import evaluate_strategy
from core.export import (
    export_summary_to_dataframe,
    export_summary_to_csv,
    export_detailed_to_json,
    export_per_question_to_csv,
)
from core.chunkers import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    SemanticChunker,
    ParentChildChunker,
)


def test_end_to_end_demo_run():
    doc = load_demo_document()
    assert len(doc) > 500, "Demo document must be non-empty and substantial"

    eval_qs = load_evaluation_questions()
    assert len(eval_qs) == 10, "Default evaluation set should have 10 labeled questions"

    embedder = FastTfidfEmbedder()

    strategies = [
        ("Fixed-Size", FixedSizeChunker(chunk_size=500, chunk_overlap=50)),
        ("Recursive", RecursiveChunker(chunk_size=500, chunk_overlap=50)),
        ("Sentence", SentenceChunker(sentences_per_chunk=3, sentence_overlap=1)),
        ("Semantic", SemanticChunker(embedder=embedder, similarity_threshold=0.6, max_chunk_size=800)),
        ("Parent-Child", ParentChildChunker(parent_chunk_size=1000, child_chunk_size=250)),
    ]

    reports = []
    for name, chunker in strategies:
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0, f"Strategy {name} failed to generate chunks"

        index = VectorIndex(chunks, embedder)
        report = evaluate_strategy(
            strategy_name=name,
            chunks=chunks,
            index=index,
            eval_questions=eval_qs,
            top_k=3,
        )
        reports.append(report)
        assert report.measured.total_chunks == len(chunks)
        assert report.measured.recall_at_k > 0.0

    # Test exports
    df = export_summary_to_dataframe(reports)
    assert len(df) == 5
    assert "Recall@K [MEASURED]" in df.columns
    assert "Total Est. Cost ($) [ESTIMATED]" in df.columns

    csv_data = export_summary_to_csv(reports)
    assert "Strategy,Top-K" in csv_data

    json_data = export_detailed_to_json(reports)
    assert "measured_results" in json_data
    assert "estimated_costs" in json_data

    query_csv = export_per_question_to_csv(reports)
    assert len(query_csv.splitlines()) == (5 * 10) + 1  # 5 strategies * 10 queries + header
