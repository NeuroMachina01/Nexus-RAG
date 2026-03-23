# Nexus-RAG — Agentic Financial Research Engine

> Hybrid retrieval (BM25 + FAISS) · Semantic caching · LangGraph agentic pipeline · RAGAS evaluation

---

## What it does

Nexus-RAG is a production-grade Retrieval Augmented Generation (RAG) system for financial research. You paste news article URLs, ask questions about them, and get grounded, hallucination-resistant answers — with sources.

It goes beyond a basic RAG pipeline in three ways:

- **Hybrid retrieval** — combines BM25 keyword search and FAISS semantic search via Reciprocal Rank Fusion, so it handles both exact financial terms ("NVDA Q3 EPS") and conceptual queries ("what is the Fed's stance on inflation")
- **Semantic caching** — similar queries return instantly from cache without calling the LLM, reducing redundant API calls
- **Agentic orchestration** — LangGraph state machine routes queries through cache → retrieval → generation, with clean separation of concerns
- **Negative constraint prompting** — the LLM is explicitly instructed what it must NOT do, driving down hallucinations

---

## Demo

```
URL input  →  Process URLs  →  Ask: "What did the Fed announce about rates?"
                                ↓
                       [Cache miss → hybrid retrieve → generate]
                                ↓
           Answer: "The Fed held rates steady at..." + Sources
                                ↓
           Ask same question again → ⚡ Served from cache instantly
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│         URL inputs │ Query box │ Metrics dashboard          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  LangGraph     │
                    │  State Machine │
                    └───────┬────────┘
                            │
              ┌─────────────▼─────────────┐
              │      check_cache node      │
              │  cosine similarity ≥ 0.85? │
              └──────┬──────────┬──────────┘
                     │ HIT      │ MISS
              ┌──────▼───┐  ┌───▼──────────────────────┐
              │ Return   │  │      retrieve node        │
              │ instantly│  │                           │
              └──────────┘  │  ┌─────────┐ ┌─────────┐ │
                            │  │  BM25   │ │  FAISS  │ │
                            │  │ sparse  │ │  dense  │ │
                            │  └────┬────┘ └────┬────┘ │
                            │       └─────┬──────┘      │
                            │        RRF merge           │
                            │    (0.4×BM25 + 0.6×FAISS) │
                            └──────────────┬─────────────┘
                                           │
                                  ┌────────▼────────┐
                                  │  generate node  │
                                  │  Negative const │
                                  │  prompt → LLM   │
                                  └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │  Store in cache │
                                  │  Return answer  │
                                  └─────────────────┘
```

### Component map

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, model init, session state, metrics dashboard |
| `rag_graph.py` | LangGraph state machine — cache → retrieve → generate flow |
| `hybrid_retriever.py` | BM25 + FAISS ensemble with weighted RRF merge and latency tracking |
| `semantic_cache.py` | Cosine similarity cache with hit rate tracking and JSON persistence |
| `evaluator.py` | RAGAS faithfulness + answer relevancy evaluation |
| `config.py` | All hyperparameters in one place |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3 via Ollama (local, no API key needed) |
| Embeddings | nomic-embed-text via Ollama |
| Agentic orchestration | LangGraph |
| Sparse retrieval | BM25 (rank_bm25) |
| Dense retrieval | FAISS (faiss-cpu) |
| Retrieval fusion | Reciprocal Rank Fusion (custom implementation) |
| Semantic cache | Cosine similarity over stored embeddings (numpy) |
| Evaluation | RAGAS (faithfulness + answer relevancy) |
| Document loading | LangChain WebBaseLoader |
| Chunking | RecursiveCharacterTextSplitter (1000 chars, 150 overlap) |
| UI | Streamlit |

---

## Prerequisites

Before running this project you need:

1. **Python 3.10+** — check with `python --version`
2. **Ollama** installed and running — download from [ollama.com](https://ollama.com)
3. **Git** (optional, for cloning)

---

## Installation

### Step 1 — Get the code

Either clone or download the ZIP and extract it:

```bash
git clone https://github.com/YOUR_USERNAME/nexus-rag.git
cd nexus-rag
```

Or just place all 7 files in a folder called `nexus-rag`.

---

### Step 2 — Pull the AI models

Open a terminal and run:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

This downloads ~4GB of model weights locally. Only needed once.

Verify Ollama is running:

```bash
ollama serve
```

You should see `Ollama is running` — keep this terminal open.

---

### Step 3 — Install Python dependencies

In a new terminal, navigate to the project folder:

```bash
cd nexus-rag
pip install streamlit langchain-community langchain-core langchain-text-splitters langchain-ollama langgraph faiss-cpu rank_bm25 ragas datasets numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

---

### Step 4 — Run the app

```bash
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.

---

## How to Use

### Step 1 — Load articles

In the sidebar, paste 1–3 news article URLs (Reuters, Bloomberg, CNBC, Economic Times, etc.) into the URL boxes and click **Process URLs**.

Wait for the green success message. The app will:
- Fetch and parse the articles
- Split them into 1000-character chunks with 150-character overlap
- Build a BM25 index from raw text
- Build a FAISS index from nomic-embed-text embeddings
- Save the FAISS index to disk for future reloads

---

### Step 2 — Ask questions

Type a question in the main text box and press Enter.

**Good queries to try:**
```
"What did the Federal Reserve announce?"
"What are the key risks mentioned?"
"Summarise the main findings"
"What is the outlook for interest rates?"
```

The pipeline will:
1. Check if a semantically similar question was asked before (cache check)
2. If cache hit → return instantly, no LLM call
3. If cache miss → run hybrid retrieval → generate answer with negative constraint prompting

---

### Step 3 — Read the metrics bar

The top of the app shows four live metrics:

| Metric | What it means |
|---|---|
| Cache hit rate | % of queries served from cache — target >30% after 20+ queries |
| Total queries | Running count of all questions asked |
| Avg retrieval (ms) | Mean hybrid retrieval latency across all calls |
| Cached entries | Number of query-answer pairs stored in cache |

---

### Step 4 — Run RAGAS evaluation (optional)

After asking 5+ questions, click **Run RAGAS evaluation** in the sidebar.

This runs the RAGAS faithfulness and answer relevancy metrics over all your Q&A pairs and shows scores between 0 and 1. A faithfulness score above 0.85 means the model is answering from context, not hallucinating.

---

## Configuration

All hyperparameters are in `config.py`:

```python
EMBEDDING_MODEL            = "nomic-embed-text"   # swap for any Ollama embedding model
LLM_MODEL                  = "llama3"             # swap for llama3.2, mistral, etc.
CACHE_SIMILARITY_THRESHOLD = 0.85                 # lower = more cache hits, lower precision
BM25_WEIGHT                = 0.4                  # weight for keyword retrieval
DENSE_WEIGHT               = 0.6                  # weight for semantic retrieval
TOP_K                      = 4                    # number of chunks to retrieve
```

### Tuning guide

**Cache threshold (0.85):** Lower it to 0.75 for more aggressive caching (more hits, occasionally wrong). Raise it to 0.92 for stricter matching (fewer hits, always correct).

**BM25/Dense weights:** For financial jargon-heavy queries, try 0.5/0.5. For conversational queries, try 0.3/0.7. Weights must not necessarily sum to 1 — they are independent multipliers in the RRF formula.

**Chunk size (1000):** Smaller chunks (500) = more precise retrieval but lose context. Larger chunks (1500) = more context but noisier retrieval.

---

## How Hybrid Retrieval Works

This is the core technical differentiator of Nexus-RAG.

### BM25 (sparse retrieval)

BM25 scores documents by term frequency with saturation and length normalisation:

```
BM25(q,d) = Σ IDF(t) × [tf(t,d) × (k₁+1)] / [tf(t,d) + k₁×(1 - b + b×|d|/avgdl)]
```

- `k₁ = 1.5` — controls how fast term frequency saturates
- `b = 0.75` — controls length normalisation penalty
- Good for: exact tickers, regulation codes, proper names, figures

### FAISS (dense retrieval)

Converts query and documents to 768-dimensional vectors using nomic-embed-text, then finds nearest neighbours by cosine similarity. Good for: paraphrased queries, conceptual questions, synonyms.

### Reciprocal Rank Fusion

Merges both ranked lists by rank position, not raw scores:

```python
score(doc) = 0.4 × (1 / rank_in_bm25) + 0.6 × (1 / rank_in_faiss)
```

Why ranks instead of scores? BM25 scores and cosine similarities are on completely different scales — they cannot be directly compared. Ranks are always comparable.

---

## How Semantic Caching Works

Every query is embedded into a 768-dim vector. On lookup:

```python
similarity = (query_embedding · cached_embedding) / (|query_emb| × |cached_emb|)
```

If `similarity ≥ 0.85` → return cached answer instantly.

This means "What did the Fed announce?" and "What was the Federal Reserve's announcement?" will hit the same cache entry — saving one full LLM inference call.

Cache entries persist to `semantic_cache.json` so they survive app restarts.

---

## How Negative Constraint Prompting Works

Standard RAG prompts say "Answer using this context." Negative constraint prompting adds explicit prohibitions:

```
STRICT RULES:
- Answer ONLY using information present in the context below.
- Do NOT use any prior knowledge, training data, or external information.
- Do NOT speculate, infer, or extrapolate beyond what the context states.
- If the context does not contain enough information, respond EXACTLY with:
  "The provided articles do not contain sufficient information..."
```

The `DO NOT` instructions directly reduce hallucinations by overriding the model's tendency to fill gaps with training knowledge.

---

## Measuring Your Resume Metrics

After running ~50 queries, open the metrics bar and record:

```
Cache hit rate  →  cache.stats["hit_rate_pct"]     (e.g. "42%")
Avg latency ms  →  hybrid.stats["avg_ms"]           (e.g. "340ms")
```

For RAGAS faithfulness:
1. Ask 10 varied questions about your loaded articles
2. Click "Run RAGAS evaluation" in sidebar
3. Record the faithfulness score (e.g. "0.91")

These three numbers are the placeholders in your resume bullets.

---

## Project Structure

```
nexus-rag/
├── app.py                  # Streamlit app — main entry point
├── rag_graph.py            # LangGraph state machine
├── hybrid_retriever.py     # BM25 + FAISS + RRF merge
├── semantic_cache.py       # Cosine similarity cache
├── evaluator.py            # RAGAS evaluation runner
├── config.py               # All hyperparameters
├── requirements.txt        # Python dependencies
├── semantic_cache.json     # Auto-created on first cache write
└── faiss_index/            # Auto-created on first URL processing
    ├── index.faiss
    └── index.pkl
```

---

## Troubleshooting

**Ollama not found / connection refused**
```bash
ollama serve   # start Ollama in a separate terminal
```

**ModuleNotFoundError for any package**
```bash
pip install --upgrade langchain langchain-core langchain-community langchain-text-splitters langchain-ollama langgraph
```

**FAISS index not found on reload**
Process URLs at least once per session. The FAISS index is saved to disk but BM25 is rebuilt in memory from the original docs each session.

**RAGAS evaluation slow**
Normal — RAGAS makes multiple LLM calls per Q&A pair. 10 pairs takes ~5 minutes on local Llama 3.

**Cache not hitting**
Lower `CACHE_SIMILARITY_THRESHOLD` in `config.py` from 0.85 to 0.78 and try again.

---

## Roadmap

- [ ] Fill resume metric placeholders with real measured numbers
- [ ] Add support for PDF document ingestion
- [ ] Add multi-document cross-referencing
- [ ] Add streaming LLM responses to UI
- [ ] Add conversation memory across turns
- [ ] Containerise with Docker for one-command deployment
- [ ] Add OpenAI/Anthropic API support as LLM backend option

---

## Author

**Puskar Sarkar**
Pre-final year B.Tech, Industrial & Production Engineering, NIT Jalandhar
[LinkedIn](https://linkedin.com/in/puskar-sarkar) · [GitHub](https://github.com/NeuroMachina01)

---

## License

MIT License — free to use, modify, and distribute.
