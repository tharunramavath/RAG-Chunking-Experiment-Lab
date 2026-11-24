"""
Evaluation Engine for RAG Chunking Experiments.
Computes real, non-fabricated retrieval metrics (Recall@K, MRR, Precision@K),
context-token counts, measured execution latencies, and explicit cost estimates.
Strictly separates measured experimental outcomes from model-based cost estimates.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from core.chunkers.base import Chunk, count_tokens
from core.retriever import VectorIndex, QueryRetrievalOutput


@dataclass
class EvalQuestion:
    """A labeled ground-truth evaluation question."""
    id: str
    question: str
    ground_truth_answers: List[str]
    relevant_keywords: Optional[List[str]] = None
    expected_section: Optional[str] = None


@dataclass
class QuestionResult:
    """Individual query outcome within an experiment."""
    question_id: str
    question: str
    hit: bool
    first_hit_rank: Optional[int]
    reciprocal_rank: float
    precision_at_k: float
    context_tokens: int
    retrieval_time_ms: float
    retrieved_chunk_ids: List[str]
    matched_ground_truth: Optional[str] = None


@dataclass
class MeasuredResults:
    """
    STRICTLY MEASURED values from the live experiment execution.
    Contains no estimates or hypothetical figures.
    """
    total_chunks: int
    mean_chunk_chars: float
    mean_chunk_tokens: float
    min_chunk_tokens: int
    max_chunk_tokens: int
    chunking_time_ms: float
    indexing_time_ms: float
    mean_retrieval_time_ms: float
    total_evaluation_questions: int
    recall_at_k: float
    mrr: float
    mean_precision_at_k: float
    total_context_tokens: int
    mean_context_tokens_per_query: float


@dataclass
class EstimatedResults:
    """
    ESTIMATED metrics computed from standard external pricing models.
    Transparently displays the underlying formulas and unit assumptions.
    """
    embedding_price_per_million_tokens: float
    context_price_per_million_tokens: float
    total_embedded_tokens: int
    estimated_embedding_cost_usd: float
    estimated_context_cost_usd: float
    total_estimated_cost_usd: float
    pricing_model_reference: str = "Industry standard (e.g. text-embedding-3-small @ $0.02/1M, prompt input @ $0.15/1M)"


@dataclass
class StrategyEvaluationReport:
    """Complete evaluation report for a single chunking strategy run."""
    strategy_name: str
    config: Dict[str, Any]
    top_k: int
    embedding_model: str
    measured: MeasuredResults
    estimated: EstimatedResults
    per_question_results: List[QuestionResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "config": self.config,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "measured_results": asdict(self.measured),
            "estimated_costs": asdict(self.estimated),
            "per_question_results": [asdict(q) for q in self.per_question_results],
        }


def is_chunk_relevant(text: str, ground_truth_answers: List[str], keywords: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """Check if chunk text satisfies ground truth answer match."""
    normalized_text = text.lower()
    for ans in ground_truth_answers:
        if ans.lower() in normalized_text:
            return True, ans
    if keywords:
        # Match if all or major keywords are present
        matches = [kw for kw in keywords if kw.lower() in normalized_text]
        if len(matches) == len(keywords):
            return True, f"keywords:{','.join(matches)}"
    return False, None


def evaluate_strategy(
    strategy_name: str,
    chunks: List[Chunk],
    index: VectorIndex,
    eval_questions: List[EvalQuestion],
    top_k: int = 3,
    chunking_time_ms: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
    embedding_price_per_1m: float = 0.02,
    context_price_per_1m: float = 0.15,
) -> StrategyEvaluationReport:
    """
    Run full fair evaluation over the evaluation questions using the prepared VectorIndex.
    Computes measured metrics and clearly separated cost estimates.
    """
    config = config or {}
    total_q = len(eval_questions)

    if not chunks or total_q == 0:
        empty_measured = MeasuredResults(
            total_chunks=len(chunks),
            mean_chunk_chars=0.0,
            mean_chunk_tokens=0.0,
            min_chunk_tokens=0,
            max_chunk_tokens=0,
            chunking_time_ms=chunking_time_ms,
            indexing_time_ms=index.indexing_time_ms,
            mean_retrieval_time_ms=0.0,
            total_evaluation_questions=total_q,
            recall_at_k=0.0,
            mrr=0.0,
            mean_precision_at_k=0.0,
            total_context_tokens=0,
            mean_context_tokens_per_query=0.0,
        )
        empty_estimated = EstimatedResults(
            embedding_price_per_million_tokens=embedding_price_per_1m,
            context_price_per_million_tokens=context_price_per_1m,
            total_embedded_tokens=0,
            estimated_embedding_cost_usd=0.0,
            estimated_context_cost_usd=0.0,
            total_estimated_cost_usd=0.0,
        )
        return StrategyEvaluationReport(
            strategy_name=strategy_name,
            config=config,
            top_k=top_k,
            embedding_model=index.embedder.model_name,
            measured=empty_measured,
            estimated=empty_estimated,
            per_question_results=[],
        )

    # Chunk level stats
    chunk_chars = [c.char_count for c in chunks]
    chunk_tokens = [c.token_count for c in chunks]
    mean_chars = float(np.mean(chunk_chars))
    mean_tokens = float(np.mean(chunk_tokens))
    min_tokens = int(np.min(chunk_tokens))
    max_tokens = int(np.max(chunk_tokens))

    # Evaluate each question
    per_q_results: List[QuestionResult] = []
    hits_count = 0
    reciprocal_ranks = []
    precisions = []
    total_context_tokens = 0
    retrieval_times = []
    total_query_tokens = 0

    for eq in eval_questions:
        q_tokens = count_tokens(eq.question)
        total_query_tokens += q_tokens

        retrieval_output: QueryRetrievalOutput = index.query(eq.question, top_k=top_k)
        retrieval_times.append(retrieval_output.retrieval_time_ms)
        total_context_tokens += retrieval_output.total_context_tokens

        hit = False
        first_hit_rank = None
        matched_gt = None
        relevant_in_top_k = 0

        for res in retrieval_output.top_k_results:
            is_rel, match = is_chunk_relevant(
                res.context_text, eq.ground_truth_answers, eq.relevant_keywords
            )
            if is_rel:
                relevant_in_top_k += 1
                if not hit:
                    hit = True
                    first_hit_rank = res.rank
                    matched_gt = match

        if hit:
            hits_count += 1
            rr = 1.0 / first_hit_rank
        else:
            rr = 0.0

        p_at_k = relevant_in_top_k / max(1, len(retrieval_output.top_k_results))
        reciprocal_ranks.append(rr)
        precisions.append(p_at_k)

        per_q_results.append(
            QuestionResult(
                question_id=eq.id,
                question=eq.question,
                hit=hit,
                first_hit_rank=first_hit_rank,
                reciprocal_rank=rr,
                precision_at_k=p_at_k,
                context_tokens=retrieval_output.total_context_tokens,
                retrieval_time_ms=retrieval_output.retrieval_time_ms,
                retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieval_output.top_k_results],
                matched_ground_truth=matched_gt,
            )
        )

    # Measured outcomes
    recall_at_k = float(hits_count / total_q) if total_q > 0 else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_retrieval_time = float(np.mean(retrieval_times)) if retrieval_times else 0.0
    mean_ctx_tokens = float(total_context_tokens / total_q) if total_q > 0 else 0.0

    measured = MeasuredResults(
        total_chunks=len(chunks),
        mean_chunk_chars=mean_chars,
        mean_chunk_tokens=mean_tokens,
        min_chunk_tokens=min_tokens,
        max_chunk_tokens=max_tokens,
        chunking_time_ms=chunking_time_ms,
        indexing_time_ms=index.indexing_time_ms,
        mean_retrieval_time_ms=mean_retrieval_time,
        total_evaluation_questions=total_q,
        recall_at_k=recall_at_k,
        mrr=mrr,
        mean_precision_at_k=mean_precision,
        total_context_tokens=total_context_tokens,
        mean_context_tokens_per_query=mean_ctx_tokens,
    )

    # Estimated costs
    # Total tokens indexed into embeddings = sum of retrieval_text tokens + all query tokens
    embedded_doc_tokens = sum(count_tokens(c.retrieval_text) for c in chunks)
    total_embedded_tokens = embedded_doc_tokens + total_query_tokens

    est_embedding_cost = (total_embedded_tokens / 1_000_000.0) * embedding_price_per_1m
    est_context_cost = (total_context_tokens / 1_000_000.0) * context_price_per_1m
    total_est_cost = est_embedding_cost + est_context_cost

    estimated = EstimatedResults(
        embedding_price_per_million_tokens=embedding_price_per_1m,
        context_price_per_million_tokens=context_price_per_1m,
        total_embedded_tokens=total_embedded_tokens,
        estimated_embedding_cost_usd=est_embedding_cost,
        estimated_context_cost_usd=est_context_cost,
        total_estimated_cost_usd=total_est_cost,
    )

    return StrategyEvaluationReport(
        strategy_name=strategy_name,
        config=config,
        top_k=top_k,
        embedding_model=index.embedder.model_name,
        measured=measured,
        estimated=estimated,
        per_question_results=per_q_results,
    )
