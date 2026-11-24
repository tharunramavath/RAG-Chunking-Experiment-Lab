"""
Export utilities for RAG Chunking Experiment results (CSV and JSON).
"""

import json
import pandas as pd
from typing import List, Dict, Any
from core.evaluation import StrategyEvaluationReport


def export_summary_to_dataframe(reports: List[StrategyEvaluationReport]) -> pd.DataFrame:
    """Format multiple strategy evaluation reports into a clean pandas DataFrame."""
    rows = []
    for r in reports:
        rows.append({
            "Strategy": r.strategy_name,
            "Top-K": r.top_k,
            "Embedding Model": r.embedding_model,
            # MEASURED METRICS
            "Total Chunks [MEASURED]": r.measured.total_chunks,
            "Mean Chunk Chars [MEASURED]": round(r.measured.mean_chunk_chars, 1),
            "Mean Chunk Tokens [MEASURED]": round(r.measured.mean_chunk_tokens, 1),
            "Recall@K [MEASURED]": round(r.measured.recall_at_k, 4),
            "MRR [MEASURED]": round(r.measured.mrr, 4),
            "Precision@K [MEASURED]": round(r.measured.mean_precision_at_k, 4),
            "Total Context Tokens [MEASURED]": r.measured.total_context_tokens,
            "Mean Context Tokens/Query [MEASURED]": round(r.measured.mean_context_tokens_per_query, 1),
            "Chunking Time (ms) [MEASURED]": round(r.measured.chunking_time_ms, 2),
            "Indexing Time (ms) [MEASURED]": round(r.measured.indexing_time_ms, 2),
            "Mean Retrieval Latency (ms) [MEASURED]": round(r.measured.mean_retrieval_time_ms, 2),
            # ESTIMATED COSTS
            "Est. Embedding Cost ($) [ESTIMATED]": f"${r.estimated.estimated_embedding_cost_usd:.6f}",
            "Est. Context Prompt Cost ($) [ESTIMATED]": f"${r.estimated.estimated_context_cost_usd:.6f}",
            "Total Est. Cost ($) [ESTIMATED]": f"${r.estimated.total_estimated_cost_usd:.6f}",
        })
    return pd.DataFrame(rows)


def export_summary_to_csv(reports: List[StrategyEvaluationReport]) -> str:
    """Generate CSV string of summary comparison."""
    df = export_summary_to_dataframe(reports)
    return df.to_csv(index=False)


def export_detailed_to_json(reports: List[StrategyEvaluationReport]) -> str:
    """Generate comprehensive JSON string with full experiment details."""
    payload = {
        "lab_experiment": "RAG Chunking Strategies Benchmark",
        "timestamp_generated": pd.Timestamp.now().isoformat(),
        "strategies_evaluated": len(reports),
        "results": [r.to_dict() for r in reports],
    }
    return json.dumps(payload, indent=2)


def export_per_question_to_csv(reports: List[StrategyEvaluationReport]) -> str:
    """Generate CSV string of query-level granular retrieval logs across all strategies."""
    rows = []
    for r in reports:
        for q in r.per_question_results:
            rows.append({
                "Strategy": r.strategy_name,
                "Question ID": q.question_id,
                "Question": q.question,
                "Hit [MEASURED]": q.hit,
                "First Hit Rank [MEASURED]": q.first_hit_rank if q.first_hit_rank else "N/A",
                "Reciprocal Rank [MEASURED]": round(q.reciprocal_rank, 4),
                "Precision@K [MEASURED]": round(q.precision_at_k, 4),
                "Context Tokens [MEASURED]": q.context_tokens,
                "Retrieval Latency (ms) [MEASURED]": round(q.retrieval_time_ms, 2),
                "Matched Target": q.matched_ground_truth or "None",
                "Retrieved Chunk IDs": "; ".join(q.retrieved_chunk_ids),
            })
    return pd.DataFrame(rows).to_csv(index=False)
