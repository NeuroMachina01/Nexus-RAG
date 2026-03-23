from typing import TypedDict, List
from langgraph.graph import StateGraph, END


# ──────────────────────────────────────────────────────────────────────────────
# NEGATIVE CONSTRAINT PROMPT
# The key technique that reduces hallucinations:
# explicitly tell the model what it MUST NOT do.
# ──────────────────────────────────────────────────────────────────────────────
NEGATIVE_CONSTRAINT_PROMPT = """\
You are a precise financial research assistant. You have been given a set of \
article excerpts as context.

STRICT RULES (do not violate any of these):
- Answer ONLY using information present in the context below.
- Do NOT use any prior knowledge, training data, or external information.
- Do NOT speculate, infer, or extrapolate beyond what the context states.
- Do NOT say "I think", "probably", "likely", "it seems", or similar hedges.
- If the context does not contain enough information to answer the question,
  respond EXACTLY with:
  "The provided articles do not contain sufficient information to answer \
this question."

---
CONTEXT:
{context}
---

QUESTION: {question}

ANSWER (strictly from context only):"""


# ──────────────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:      str
    docs:          List
    context:       str
    answer:        str
    sources:       str
    cache_hit:     bool
    retrieval_ms:  float


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ──────────────────────────────────────────────────────────────────────────────
def build_rag_graph(llm, hybrid_retriever, cache):
    """
    Compile the LangGraph pipeline.

    Flow:
      check_cache
        ├─ hit  ──► END                (instant, no LLM call)
        └─ miss ──► retrieve ──► generate ──► END
    """

    # ── Node 1: check cache ──────────────────────────────────────────────────
    def check_cache_node(state: RAGState) -> RAGState:
        cached = cache.lookup(state["question"])
        if cached:
            return {
                **state,
                "answer":    cached["answer"],
                "sources":   cached["sources"],
                "cache_hit": True,
            }
        return {**state, "cache_hit": False}

    # ── Node 2: hybrid retrieve ──────────────────────────────────────────────
    def retrieve_node(state: RAGState) -> RAGState:
        import time
        start = time.perf_counter()
        docs  = hybrid_retriever.retrieve(state["question"])
        ms    = (time.perf_counter() - start) * 1_000

        context = "\n\n---\n\n".join(d.page_content for d in docs)
        sources = "\n".join(
            sorted({d.metadata.get("source", "") for d in docs if d.metadata.get("source")})
        )
        return {**state, "docs": docs, "context": context,
                "sources": sources, "retrieval_ms": ms}

    # ── Node 3: generate with negative constraint prompt ─────────────────────
    def generate_node(state: RAGState) -> RAGState:
        prompt = NEGATIVE_CONSTRAINT_PROMPT.format(
            context=state["context"],
            question=state["question"],
        )
        answer = llm.invoke(prompt)
        # Store in cache for future similar queries
        cache.store(state["question"], answer, state["sources"])
        return {**state, "answer": answer}

    # ── Routing ──────────────────────────────────────────────────────────────
    def route_on_cache(state: RAGState) -> str:
        return "end" if state["cache_hit"] else "retrieve"

    # ── Compile ──────────────────────────────────────────────────────────────
    graph = StateGraph(RAGState)
    graph.add_node("check_cache", check_cache_node)
    graph.add_node("retrieve",    retrieve_node)
    graph.add_node("generate",    generate_node)

    graph.set_entry_point("check_cache")
    graph.add_conditional_edges(
        "check_cache",
        route_on_cache,
        {"end": END, "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
