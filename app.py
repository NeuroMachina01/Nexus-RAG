import os
import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

from config import VECTORSTORE_DIR, EMBEDDING_MODEL, LLM_MODEL
from semantic_cache import SemanticCache
from hybrid_retriever import HybridRetriever
from rag_graph import build_rag_graph


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexus-RAG: Financial Research Tool",
    page_icon="📈",
    layout="wide",
)
st.title("📈 Nexus-RAG — Financial Research Tool")
st.caption("Hybrid retrieval (BM25 + FAISS) · Semantic caching · Agentic pipeline (LangGraph)")


# ──────────────────────────────────────────────────────────────────────────────
# MODELS  (cached so Streamlit doesn't reload on every rerun)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    llm        = OllamaLLM(model=LLM_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return llm, embeddings

llm, embeddings = load_models()


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────
if "cache"    not in st.session_state:
    st.session_state.cache    = SemanticCache(embeddings)
if "hybrid"   not in st.session_state:
    st.session_state.hybrid   = None   # built after URLs are processed
if "graph"    not in st.session_state:
    st.session_state.graph    = None
if "qa_log"   not in st.session_state:
    st.session_state.qa_log   = []     # list of {question, answer, contexts, retrieval_ms}


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — URL INPUT + PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Step 1 — Load articles")
    urls = [st.text_input(f"URL {i+1}", key=f"url_{i}") for i in range(3)]
    urls = [u.strip() for u in urls if u.strip()]

    process_btn = st.button("Process URLs", type="primary", use_container_width=True)

    if process_btn:
        if not urls:
            st.error("Enter at least one URL.")
        else:
            with st.spinner("Loading & indexing articles…"):
                try:
                    # Load
                    docs_raw = WebBaseLoader(urls).load()

                    # Chunk
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150,
                        separators=["\n\n", "\n", ".", ","],
                    )
                    docs = splitter.split_documents(docs_raw)

                    # Build FAISS vectorstore
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    vectorstore.save_local(VECTORSTORE_DIR)

                    # Build hybrid retriever (needs raw docs for BM25)
                    st.session_state.hybrid = HybridRetriever(docs, vectorstore)
                    st.session_state.docs   = docs

                    # Build LangGraph pipeline
                    st.session_state.graph = build_rag_graph(
                        llm,
                        st.session_state.hybrid,
                        st.session_state.cache,
                    )
                    st.success(f"Indexed {len(docs)} chunks from {len(urls)} article(s).")
                except Exception as e:
                    st.error(f"Processing error: {e}")

    st.divider()

    # ── Reload from saved index ────────────────────────────────────────────
    st.header("Or reload saved index")
    if st.button("Load saved index", use_container_width=True):
        if not os.path.exists(VECTORSTORE_DIR):
            st.warning("No saved index found. Process URLs first.")
        elif "docs" not in st.session_state:
            st.warning("BM25 needs the original docs. Process URLs at least once per session.")
        else:
            vectorstore = FAISS.load_local(
                VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
            )
            st.session_state.hybrid = HybridRetriever(
                st.session_state.docs, vectorstore
            )
            st.session_state.graph = build_rag_graph(
                llm, st.session_state.hybrid, st.session_state.cache
            )
            st.success("Index reloaded.")

    st.divider()

    # ── RAGAS eval launcher ────────────────────────────────────────────────
    st.header("Step 3 — Evaluate (RAGAS)")
    st.caption("Run after collecting ≥5 Q&A pairs.")
    if st.button("Run RAGAS evaluation", use_container_width=True):
        log = st.session_state.qa_log
        if len(log) < 2:
            st.warning("Need at least 2 Q&A pairs first.")
        else:
            try:
                from evaluator import run_ragas_eval
                scores = run_ragas_eval(
                    questions  = [e["question"] for e in log],
                    answers    = [e["answer"]   for e in log],
                    contexts   = [e["contexts"]  for e in log],
                    llm        = llm,
                    embeddings = embeddings,
                )
                st.success("Evaluation complete!")
                st.metric("Faithfulness",     scores["faithfulness"])
                st.metric("Answer relevancy", scores["answer_relevancy"])
            except Exception as e:
                st.error(f"RAGAS error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — METRICS DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
cache_stats   = st.session_state.cache.stats
hybrid_stats  = st.session_state.hybrid.stats if st.session_state.hybrid else {}

col1.metric("Cache hit rate",     f"{cache_stats['hit_rate_pct']}%",
            help="% of queries served from semantic cache (your resume metric)")
col2.metric("Total queries",      cache_stats["total_queries"])
col3.metric("Avg retrieval (ms)", hybrid_stats.get("avg_ms", "—"),
            help="Mean hybrid retrieval latency across all calls (your resume metric)")
col4.metric("Cached entries",     cache_stats["cached_entries"])

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — QUESTION ANSWERING
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Ask a question")
query = st.text_input("Question based on the loaded articles:", placeholder="e.g. What did the Fed announce about interest rates?")

if query:
    if st.session_state.graph is None:
        st.warning("Process URLs first (sidebar → Step 1).")
    else:
        with st.spinner("Running pipeline…"):
            try:
                initial_state = {
                    "question":     query,
                    "docs":         [],
                    "context":      "",
                    "answer":       "",
                    "sources":      "",
                    "cache_hit":    False,
                    "retrieval_ms": 0.0,
                }
                result = st.session_state.graph.invoke(initial_state)

                # Log for RAGAS evaluation later
                contexts_for_eval = [d.page_content for d in result.get("docs", [])]
                st.session_state.qa_log.append({
                    "question": query,
                    "answer":   result["answer"],
                    "contexts": contexts_for_eval,
                })

                # ── Answer display ─────────────────────────────────────────
                if result.get("cache_hit"):
                    st.info("⚡ Served from semantic cache — no LLM call made.")

                st.markdown("#### Answer")
                st.write(result["answer"])

                if result.get("retrieval_ms"):
                    st.caption(f"Retrieval latency: {result['retrieval_ms']:.1f} ms")

                if result.get("sources"):
                    with st.expander("Sources"):
                        for src in result["sources"].split("\n"):
                            if src.strip():
                                st.markdown(f"- {src.strip()}")

                # Refresh metrics
                st.rerun()

            except Exception as e:
                st.error(f"Pipeline error: {e}")
