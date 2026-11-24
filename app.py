"""
RAG Chunking Experiment Lab
A reproducible benchmark environment to evaluate chunking strategies
under strictly controlled, fair retrieval conditions.
"""

import time
import json
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

from core.chunkers import (
    BaseChunker,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    SemanticChunker,
    ParentChildChunker,
    count_tokens,
)
from core.embeddings import get_embedder, BaseEmbedder
from core.retriever import VectorIndex
from core.evaluation import evaluate_strategy, EvalQuestion, StrategyEvaluationReport
from core.dataset import load_demo_document, load_evaluation_questions
from core.document_loader import load_text_from_file
from core.export import (
    export_summary_to_dataframe,
    export_summary_to_csv,
    export_detailed_to_json,
    export_per_question_to_csv,
)

# Page configuration
st.set_page_config(
    page_title="RAG Chunking Experiment Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern Cream & Red theme with polished typography and component styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main App Background - Rich Warm Cream */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F6F1E8 !important;
    }
    .main .block-container {
        background-color: transparent !important;
    }
    [data-testid="stSidebar"] {
        background-color: #EDE6DA !important;
        border-right: 1px solid #DFD5C5 !important;
    }

    /* Hero Brand Banner */
    .hero-container {
        background: linear-gradient(135deg, #FFFDF9 0%, #FAF5ED 100%) !important;
        border: 1px solid #E4DACB !important;
        border-left: 5px solid #C92A2A !important;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(201, 42, 42, 0.04);
    }
    .hero-badge {
        display: inline-block;
        background-color: #FFF0F0;
        color: #C92A2A;
        border: 1px solid #FFD0D0;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 12px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #231F20;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #5C524F;
        margin: 0;
        line-height: 1.5;
    }

    /* Measured and Estimated Badges */
    .metric-badge-measured {
        background: linear-gradient(135deg, #C92A2A 0%, #A61E1E 100%);
        color: #FFFFFF;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        box-shadow: 0 2px 5px rgba(201, 42, 42, 0.2);
    }
    .metric-badge-estimated {
        background: linear-gradient(135deg, #6C5E57 0%, #4E443F 100%);
        color: #FFFFFF;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.12);
    }

    /* Clean Card Containers */
    .cream-card {
        background-color: #FFFFFF;
        border: 1px solid #ECE3D6;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(35, 31, 32, 0.03);
    }
    .chunk-card {
        border-left: 4px solid #C92A2A;
        background-color: #FFFDFB;
        border-top: 1px solid #EFE6DA;
        border-right: 1px solid #EFE6DA;
        border-bottom: 1px solid #EFE6DA;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    }

    /* Metric widget polish */
    div[data-testid="stMetric"] {
        background-color: #FFFDF9 !important;
        border: 1px solid #E4DACB !important;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #FFC9C9 !important;
        box-shadow: 0 4px 12px rgba(201, 42, 42, 0.08);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        color: #5C524F;
        letter-spacing: 0.02em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 800;
        color: #231F20;
    }

    /* Tab Polish */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #5C524F !important;
        font-size: 0.95rem;
        padding-top: 10px;
        padding-bottom: 12px;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #C92A2A !important;
        border-bottom-color: #C92A2A !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #C92A2A !important;
    }

    /* Primary and Secondary Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #C92A2A 0%, #A61E1E 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        border-radius: 8px !important;
        box-shadow: 0 3px 10px rgba(201, 42, 42, 0.25) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 14px rgba(201, 42, 42, 0.35) !important;
    }
    button[kind="secondary"] {
        border: 1px solid #E2D7C8 !important;
        background-color: #FFFFFF !important;
        color: #231F20 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease !important;
    }
    button[kind="secondary"]:hover {
        border-color: #C92A2A !important;
        color: #C92A2A !important;
        background-color: #FFFDFB !important;
    }

    /* Sidebar Header styling */
    [data-testid="stSidebar"] {
        background-color: #F8F4EC;
        border-right: 1px solid #ECE3D6;
    }

    /* Expander styling */
    details {
        border: 1px solid #E4DACB !important;
        border-radius: 8px !important;
        background-color: #FFFDF9 !important;
        margin-bottom: 10px !important;
    }
    summary {
        font-weight: 600 !important;
        color: #231F20 !important;
    }

    /* Input & Select fields on warm cream */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #FFFDF9 !important;
        border-color: #DFD5C5 !important;
    }
    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {
        border-color: #C92A2A !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_cached_embedder(model_choice: str) -> BaseEmbedder:
    """Load and cache embedding model so it is not re-instantiated on every interaction."""
    return get_embedder(model_choice)


# Session state initialization
if "document_text" not in st.session_state:
    st.session_state.document_text = load_demo_document()
if "eval_questions" not in st.session_state:
    st.session_state.eval_questions = load_evaluation_questions()
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []
if "last_chunk_strategy" not in st.session_state:
    st.session_state.last_chunk_strategy = ""
if "benchmark_reports" not in st.session_state:
    st.session_state.benchmark_reports = []


# ==========================================
# SIDEBAR: Controlled / Fixed Experiment Setup
# ==========================================
st.sidebar.title("🔬 Experiment Controls")
st.sidebar.markdown(
    """
    <div style="background-color: #FFFDFB; border: 1px solid #EFE6DA; border-left: 4px solid #C92A2A; padding: 12px 14px; border-radius: 8px; margin-bottom: 16px;">
        <strong style="color: #C92A2A; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;">Fair Benchmark Guarantee</strong><br/>
        <span style="font-size: 0.82rem; color: #5C524F; line-height: 1.4;">
            Embedding model, dataset, retrieval metric, and Top-K are <b>held strictly constant</b> across strategies.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Embedding Model
embedder_options = [
    "all-MiniLM-L6-v2",
    "tfidf-dense-local",
]
selected_embedder_name = st.sidebar.selectbox(
    "Fixed Embedding Model",
    embedder_options,
    index=0,
    help="Both dense neural (MiniLM) and fast deterministic (TF-IDF) models available.",
)

# 2. Top-K Retrieval
top_k_val = st.sidebar.slider(
    "Fixed Top-K Retrieval",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of chunks retrieved for context synthesis.",
)

# 3. Cost Assumptions (Formula Parameters)
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Cost Assumptions [ESTIMATES]")
st.sidebar.caption("Configurable rate tiers used to compute estimated costs:")
cost_embed_per_1m = st.sidebar.number_input(
    "Embedding Cost ($ / 1M tokens)",
    value=0.02,
    min_value=0.0,
    step=0.005,
    format="%.4f",
    help="Default $0.02 / 1M tokens (standard text-embedding-3-small tier)",
)
cost_context_per_1m = st.sidebar.number_input(
    "Context Prompt Cost ($ / 1M tokens)",
    value=0.15,
    min_value=0.0,
    step=0.01,
    format="%.4f",
    help="Default $0.15 / 1M tokens (standard lightweight frontier model prompt tier)",
)

# Brand Hero Header
st.markdown(
    """
    <div class="hero-container">
        <span class="hero-badge">Reproducible Benchmark Suite</span>
        <h1 class="hero-title">RAG Chunking Experiment Lab</h1>
        <p class="hero-subtitle">
            Controlled empirical evaluation of document chunking strategies under strictly fixed retrieval conditions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Navigation Tabs
tab_doc, tab_preview, tab_benchmark, tab_export = st.tabs([
    "📄 1. Document & Evaluation Set",
    "⚙️ 2. Strategy Config & Preview",
    "🧪 3. Fair Benchmark Lab",
    "💾 4. Export & Reports",
])


# ==========================================
# TAB 1: Document & Evaluation Set
# ==========================================
with tab_doc:
    st.header("Document Ingestion & Labeled Ground Truth")
    col_upload, col_eval = st.columns([1, 1])

    with col_upload:
        st.subheader("Document Source")
        upload_mode = st.radio(
            "Select Input Source",
            ["Use Built-in Demo Whitepaper", "Upload PDF/TXT Document"],
            horizontal=True,
        )

        if upload_mode == "Upload PDF/TXT Document":
            uploaded_file = st.file_uploader(
                "Upload a PDF or TXT file",
                type=["txt", "pdf"],
                help="Accepts plain text and PDF documents.",
            )
            if uploaded_file is not None:
                extracted = load_text_from_file(uploaded_file, filename=uploaded_file.name)
                st.session_state.document_text = extracted
                st.success(f"Loaded '{uploaded_file.name}' successfully!")
        else:
            if st.button("Reload Built-in Demo Document"):
                st.session_state.document_text = load_demo_document()
                st.info("Demo document reloaded.")

        doc_text = st.session_state.document_text
        doc_tokens = count_tokens(doc_text)
        st.markdown(
            f"""
            **Document Statistics**:
            - Characters: `{len(doc_text):,}`
            - Estimated Tokens (`cl100k_base`): `{doc_tokens:,}`
            - Words: `{len(doc_text.split()):,}`
            """
        )
        with st.expander("Preview Ingested Document Text", expanded=False):
            st.text_area("Document Content", doc_text, height=250, disabled=True)

    with col_eval:
        st.subheader("Labeled Evaluation Questions")
        eval_mode = st.radio(
            "Evaluation Questions Source",
            ["Use Built-in Benchmark (10 Questions)", "Upload Custom JSON Questions"],
            horizontal=True,
        )

        if eval_mode == "Upload Custom JSON Questions":
            eval_file = st.file_uploader(
                "Upload JSON Evaluation File",
                type=["json"],
                help="JSON file containing a list of {id, question, ground_truth_answers, relevant_keywords}",
            )
            if eval_file is not None:
                try:
                    custom_eqs = load_evaluation_questions(eval_file.read().decode("utf-8"))
                    st.session_state.eval_questions = custom_eqs
                    st.success(f"Loaded {len(custom_eqs)} custom evaluation questions!")
                except Exception as e:
                    st.error(f"Error parsing JSON: {e}")
        else:
            if st.button("Reset to Default 10 Questions"):
                st.session_state.eval_questions = load_evaluation_questions()
                st.info("Reset to default questions.")

        questions = st.session_state.eval_questions
        st.write(f"Total Active Evaluation Queries: **{len(questions)}**")
        with st.expander("View Evaluation Questions & Ground Truth Targets", expanded=False):
            q_data = []
            for q in questions:
                q_data.append({
                    "ID": q.id,
                    "Question": q.question,
                    "Ground Truth Target Strings": ", ".join(q.ground_truth_answers),
                    "Expected Section": q.expected_section or "N/A",
                })
            st.dataframe(pd.DataFrame(q_data), use_container_width=True)


# ==========================================
# TAB 2: Strategy Config & Preview
# ==========================================
with tab_preview:
    st.header("Strategy Configuration & Chunk Inspector")
    st.markdown("Fine-tune parameters for any individual chunking strategy and preview the resulting chunks in real time.")

    strategy_choice = st.selectbox(
        "Choose Chunking Strategy",
        ["Fixed-Size", "Recursive", "Sentence", "Semantic", "Parent-Child"],
    )

    param_col1, param_col2, param_col3 = st.columns(3)

    chunker_instance: BaseChunker = None

    if strategy_choice == "Fixed-Size":
        with param_col1:
            fs_size = st.slider("Chunk Size", 100, 2000, 500, step=50)
        with param_col2:
            fs_overlap = st.slider("Chunk Overlap", 0, min(fs_size - 1, 500), 50, step=10)
        with param_col3:
            fs_unit = st.selectbox("Unit", ["characters", "tokens"])
        chunker_instance = FixedSizeChunker(chunk_size=fs_size, chunk_overlap=fs_overlap, unit=fs_unit)

    elif strategy_choice == "Recursive":
        with param_col1:
            rc_size = st.slider("Target Chunk Size (chars)", 100, 2000, 500, step=50)
        with param_col2:
            rc_overlap = st.slider("Chunk Overlap (chars)", 0, min(rc_size - 1, 500), 50, step=10)
        with param_col3:
            st.caption("Separators: `['\\n\\n', '\\n', '. ', ' ', '']`")
        chunker_instance = RecursiveChunker(chunk_size=rc_size, chunk_overlap=rc_overlap)

    elif strategy_choice == "Sentence":
        with param_col1:
            st_count = st.slider("Sentences Per Chunk", 1, 10, 3, step=1)
        with param_col2:
            st_overlap = st.slider("Sentence Overlap", 0, st_count - 1, 1, step=1)
        with param_col3:
            st_max_size = st.slider("Max Chunk Size Guard (chars)", 200, 3000, 800, step=50)
        chunker_instance = SentenceChunker(
            sentences_per_chunk=st_count,
            sentence_overlap=st_overlap,
            max_chunk_size=st_max_size,
        )

    elif strategy_choice == "Semantic":
        with param_col1:
            sem_type = st.selectbox("Threshold Type", ["percentile", "threshold"])
        with param_col2:
            if sem_type == "percentile":
                sem_perc = st.slider("Distance Percentile Cutoff", 50.0, 95.0, 75.0, step=2.5)
                sem_thresh = 0.65
            else:
                sem_thresh = st.slider("Min Cosine Similarity Cutoff", 0.1, 0.95, 0.65, step=0.05)
                sem_perc = 75.0
        with param_col3:
            sem_max = st.slider("Max Chunk Size (chars)", 300, 3000, 1000, step=50)
        # Pre-assign embedder for semantic chunker
        embedder = load_cached_embedder(selected_embedder_name)
        chunker_instance = SemanticChunker(
            embedder=embedder,
            similarity_threshold=sem_thresh,
            threshold_type=sem_type,
            threshold_percentile=sem_perc,
            max_chunk_size=sem_max,
        )

    elif strategy_choice == "Parent-Child":
        with param_col1:
            pc_parent_size = st.slider("Parent Chunk Size (chars)", 500, 3000, 1200, step=100)
            pc_parent_overlap = st.slider("Parent Overlap (chars)", 0, 400, 100, step=20)
        with param_col2:
            pc_child_size = st.slider("Child Chunk Size (chars)", 100, min(800, pc_parent_size - 50), 300, step=25)
            pc_child_overlap = st.slider("Child Overlap (chars)", 0, min(150, pc_child_size - 10), 50, step=10)
        with param_col3:
            st.info("ℹ️ Child chunks are indexed into vector space. Upon match, the full Parent chunk is retrieved for context.")
        chunker_instance = ParentChildChunker(
            parent_chunk_size=pc_parent_size,
            parent_overlap=pc_parent_overlap,
            child_chunk_size=pc_child_size,
            child_overlap=pc_child_overlap,
        )

    col_btn, col_stats = st.columns([1, 3])
    with col_btn:
        run_preview = st.button("Chunk Document & Preview", type="primary")

    if run_preview or (st.session_state.last_chunk_strategy == strategy_choice and st.session_state.last_chunks):
        t0 = time.perf_counter()
        preview_chunks = chunker_instance.chunk(st.session_state.document_text)
        chunk_time_ms = (time.perf_counter() - t0) * 1000.0
        st.session_state.last_chunks = preview_chunks
        st.session_state.last_chunk_strategy = strategy_choice

        # Statistics overview
        tokens = [c.token_count for c in preview_chunks]
        chars = [c.char_count for c in preview_chunks]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Chunks [MEASURED]", len(preview_chunks))
        m2.metric("Mean Tokens [MEASURED]", f"{np.mean(tokens):.1f}" if tokens else 0)
        m3.metric("Min Tokens [MEASURED]", f"{np.min(tokens)}" if tokens else 0)
        m4.metric("Max Tokens [MEASURED]", f"{np.max(tokens)}" if tokens else 0)
        m5.metric("Chunking Latency [MEASURED]", f"{chunk_time_ms:.2f} ms")

        # Visual Token Distribution Chart
        if tokens:
            st.subheader("Chunk Token Distribution")
            df_hist = pd.DataFrame({"Token Count": tokens})
            st.bar_chart(df_hist["Token Count"].value_counts().sort_index())

        # Individual Chunk Inspector
        st.subheader(f"Generated Chunks (First 15 of {len(preview_chunks)})")
        for i, chk in enumerate(preview_chunks[:15]):
            with st.expander(f"Chunk #{i+1} — ID: `{chk.chunk_id}` | Tokens: `{chk.token_count}` | Chars: `{chk.char_count}`"):
                st.markdown(f"**Retrieval Text (Indexed in Vector Store):**")
                st.code(chk.retrieval_text, language="markdown")
                if chk.parent_id:
                    st.markdown(f"**Parent Context Payload (Provided to Prompt on Retrieval):**")
                    st.caption(f"Parent ID: `{chk.parent_id}`")
                    st.code(chk.context_text, language="markdown")


# ==========================================
# TAB 3: Fair Benchmark Lab
# ==========================================
with tab_benchmark:
    st.header("Fair Head-to-Head Benchmark Lab")
    st.markdown(
        """
        Run all chunking strategies against the **same dataset**, the **same fixed embedding model**, 
        and the **same retrieval method and Top-K**.
        
        *Notice: All retrieval metrics, latencies, and token counts are **strictly measured** on live executions.
        Cost figures are calculated using transparent unit-rate cost formulas and designated as **estimates**.*
        """
    )

    strategies_to_benchmark = st.multiselect(
        "Select Strategies to Include in Benchmark",
        ["Fixed-Size", "Recursive", "Sentence", "Semantic", "Parent-Child"],
        default=["Fixed-Size", "Recursive", "Sentence", "Semantic", "Parent-Child"],
    )

    run_benchmark = st.button("🚀 Run Live Fair Benchmark", type="primary")

    if run_benchmark:
        if not strategies_to_benchmark:
            st.warning("Please select at least one strategy to benchmark.")
        else:
            with st.spinner(f"Loading embedding model '{selected_embedder_name}'..."):
                embedder = load_cached_embedder(selected_embedder_name)

            doc = st.session_state.document_text
            eval_qs = st.session_state.eval_questions

            benchmark_results: List[StrategyEvaluationReport] = []
            progress_bar = st.progress(0)

            for idx, strat_name in enumerate(strategies_to_benchmark):
                st.toast(f"Running strategy {idx+1}/{len(strategies_to_benchmark)}: {strat_name}...")
                
                # Instantiate chunker
                if strat_name == "Fixed-Size":
                    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50, unit="characters")
                elif strat_name == "Recursive":
                    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
                elif strat_name == "Sentence":
                    chunker = SentenceChunker(sentences_per_chunk=3, sentence_overlap=1, max_chunk_size=800)
                elif strat_name == "Semantic":
                    chunker = SemanticChunker(
                        embedder=embedder,
                        similarity_threshold=0.65,
                        threshold_type="percentile",
                        threshold_percentile=75.0,
                        max_chunk_size=1000,
                    )
                elif strat_name == "Parent-Child":
                    chunker = ParentChildChunker(
                        parent_chunk_size=1200,
                        parent_overlap=100,
                        child_chunk_size=300,
                        child_overlap=50,
                    )

                # 1. Chunking
                t_chunk0 = time.perf_counter()
                chunks = chunker.chunk(doc)
                t_chunk_ms = (time.perf_counter() - t_chunk0) * 1000.0

                # 2. Vector Indexing
                index = VectorIndex(chunks, embedder)

                # 3. Live Evaluation
                report = evaluate_strategy(
                    strategy_name=strat_name,
                    chunks=chunks,
                    index=index,
                    eval_questions=eval_qs,
                    top_k=top_k_val,
                    chunking_time_ms=t_chunk_ms,
                    config=chunker.get_config(),
                    embedding_price_per_1m=cost_embed_per_1m,
                    context_price_per_1m=cost_context_per_1m,
                )
                benchmark_results.append(report)
                progress_bar.progress((idx + 1) / len(strategies_to_benchmark))

            st.session_state.benchmark_reports = benchmark_results
            st.success("Benchmark completed successfully!")

    if st.session_state.benchmark_reports:
        reports = st.session_state.benchmark_reports
        df_summary = export_summary_to_dataframe(reports)

        st.subheader("📊 Comparative Benchmark Results")
        st.markdown(
            """
            <span class="metric-badge-measured">MEASURED</span> Actual runtime metrics (Recall@K, MRR, Context Tokens, Latencies).  
            <span class="metric-badge-estimated">ESTIMATED</span> Modeled financial costs ($) based on token usage and standard unit rates.
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(df_summary, use_container_width=True)

        # Visual Comparison Charts
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("Retrieval Quality: Recall@K vs. MRR [MEASURED]")
            chart_data = pd.DataFrame({
                "Strategy": [r.strategy_name for r in reports],
                "Recall@K": [r.measured.recall_at_k for r in reports],
                "MRR": [r.measured.mrr for r in reports],
            }).set_index("Strategy")
            st.bar_chart(chart_data)

        with col_c2:
            st.subheader("Context Tokens vs. Latency [MEASURED]")
            chart_perf = pd.DataFrame({
                "Strategy": [r.strategy_name for r in reports],
                "Mean Context Tokens/Query": [r.measured.mean_context_tokens_per_query for r in reports],
                "Mean Retrieval Latency (ms)": [r.measured.mean_retrieval_time_ms for r in reports],
            }).set_index("Strategy")
            st.bar_chart(chart_perf)

        # Cost Breakdown Comparison
        st.subheader("Estimated Cost Economics Breakdown [ESTIMATED]")
        cost_df = pd.DataFrame({
            "Strategy": [r.strategy_name for r in reports],
            "Est. Embedding Cost ($)": [r.estimated.estimated_embedding_cost_usd for r in reports],
            "Est. Context Prompt Cost ($)": [r.estimated.estimated_context_cost_usd for r in reports],
            "Total Est. Cost ($)": [r.estimated.total_estimated_cost_usd for r in reports],
        }).set_index("Strategy")
        st.bar_chart(cost_df[["Est. Embedding Cost ($)", "Est. Context Prompt Cost ($)"]])

        # Detailed Query-Level Inspector
        st.subheader("🔍 Query-Level Ground Truth Verification Inspector")
        all_q_ids = [q.id for q in st.session_state.eval_questions]
        selected_q_id = st.selectbox("Inspect Evaluation Question", all_q_ids)

        active_q = next((q for q in st.session_state.eval_questions if q.id == selected_q_id), None)
        if active_q:
            st.markdown(f"**Query**: *\"{active_q.question}\"*")
            st.markdown(f"**Target Ground Truth**: `{active_q.ground_truth_answers}`")

            q_cols = st.columns(len(reports))
            for i, r in enumerate(reports):
                with q_cols[i]:
                    q_res = next((res for res in r.per_question_results if res.question_id == selected_q_id), None)
                    if q_res:
                        if q_res.hit:
                            st.success(f"**{r.strategy_name}**: HIT ✅ (Rank {q_res.first_hit_rank})")
                            st.caption(f"Matched target: `{q_res.matched_ground_truth}`")
                        else:
                            st.error(f"**{r.strategy_name}**: MISS ❌")
                        st.caption(f"Context Tokens: {q_res.context_tokens} | Latency: {q_res.retrieval_time_ms:.2f}ms")


# ==========================================
# TAB 4: Export & Reports
# ==========================================
with tab_export:
    st.header("Export Benchmark Reports")
    st.markdown("Download full, non-fabricated reproducibility reports for audit and documentation.")

    if not st.session_state.benchmark_reports:
        st.info("No benchmark results to export yet. Please execute a benchmark in Tab 3 first.")
    else:
        reports = st.session_state.benchmark_reports

        csv_summary = export_summary_to_csv(reports)
        json_detailed = export_detailed_to_json(reports)
        csv_questions = export_per_question_to_csv(reports)

        c_exp1, c_exp2, c_exp3 = st.columns(3)

        with c_exp1:
            st.subheader("1. Summary CSV")
            st.caption("Strategy comparison table with measured metrics and estimated costs.")
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv_summary,
                file_name="rag_chunking_summary_report.csv",
                mime="text/csv",
                type="primary",
            )

        with c_exp2:
            st.subheader("2. Detailed JSON")
            st.caption("Complete schema including parameters, per-question logs, and execution timestamps.")
            st.download_button(
                label="📥 Download Detailed JSON",
                data=json_detailed,
                file_name="rag_chunking_detailed_report.json",
                mime="application/json",
            )

        with c_exp3:
            st.subheader("3. Query Log CSV")
            st.caption("Per-query hit/miss status, ranks, reciprocal rank, and chunk IDs.")
            st.download_button(
                label="📥 Download Query Log CSV",
                data=csv_questions,
                file_name="rag_chunking_query_logs.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.subheader("JSON Payload Preview")
        st.code(json_detailed[:1500] + "\n\n... (truncated preview) ...", language="json")
