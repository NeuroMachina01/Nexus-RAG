"""
RAGAS evaluation module.

Run this SEPARATELY after you have collected question/answer/context
triples from real usage. Do NOT call it on every query — it's slow
and costs extra LLM calls.

Usage:
    from evaluator import run_ragas_eval
    result = run_ragas_eval(
        questions=["What did the Fed announce?"],
        answers=["The Fed held rates steady."],
        contexts=[["...article chunk 1...", "...article chunk 2..."]],
        llm=your_llm,
        embeddings=your_embeddings,
    )
    print(result)   # → {"faithfulness": 0.91, "answer_relevancy": 0.88}
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def run_ragas_eval(
    questions: list[str],
    answers:   list[str],
    contexts:  list[list[str]],   # list of context-chunk lists, one per question
    llm,
    embeddings,
) -> dict:
    """
    Evaluates faithfulness and answer_relevancy via RAGAS.

    Returns a dict with metric scores you can log and report on your resume.

    Args:
        questions  : List of user questions
        answers    : List of model answers (from your pipeline)
        contexts   : For each question, a list of retrieved doc strings
        llm        : Your LangChain-compatible LLM
        embeddings : Your LangChain-compatible embeddings
    """
    dataset = Dataset.from_dict({
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    })

    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_emb = LangchainEmbeddingsWrapper(embeddings)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=wrapped_llm,
        embeddings=wrapped_emb,
    )

    scores = {
        "faithfulness":      round(float(result["faithfulness"]),      3),
        "answer_relevancy":  round(float(result["answer_relevancy"]),  3),
        "n_evaluated":       len(questions),
    }
    print("\n── RAGAS Evaluation Results ──")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    return scores
