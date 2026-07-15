"""Transparent retrieval metrics for the literature RAG subsystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence


def load_benchmark(path: Path) -> List[dict]:
    """Load JSONL cases with ``question`` and ``relevant_ids`` fields."""
    cases = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not case.get("question") or not case.get("relevant_ids"):
                raise ValueError(f"Invalid benchmark case on line {line_number}")
            case["relevant_ids"] = [str(value) for value in case["relevant_ids"]]
            cases.append(case)
    if not cases:
        raise ValueError("Benchmark is empty")
    return cases


def score_ranked_ids(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> dict:
    """Compute precision@k, recall@k, hit@k, and reciprocal rank."""
    relevant = {str(value) for value in relevant_ids}
    ranked = [str(value) for value in retrieved_ids[:k]]
    hits = [value for value in ranked if value in relevant]
    reciprocal_rank = 0.0
    for rank, value in enumerate(ranked, start=1):
        if value in relevant:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "precision_at_k": len(set(hits)) / max(k, 1),
        "recall_at_k": len(set(hits)) / len(relevant),
        "hit_at_k": float(bool(hits)),
        "reciprocal_rank": reciprocal_rank,
    }


def evaluate_rankings(cases: Sequence[dict], k: int = 5) -> dict:
    """Aggregate metrics for cases that already contain ``retrieved_ids``."""
    scored = []
    for case in cases:
        metrics = score_ranked_ids(case.get("retrieved_ids", []), case["relevant_ids"], k)
        scored.append({"question": case["question"], **metrics})
    keys = ["precision_at_k", "recall_at_k", "hit_at_k", "reciprocal_rank"]
    aggregate = {key: sum(case[key] for case in scored) / len(scored) for key in keys}
    return {"k": k, "cases": len(scored), "aggregate": aggregate, "per_question": scored}


def evaluate_retriever(retriever, cases: Sequence[dict], k: int = 5) -> dict:
    """Invoke a LangChain-style retriever and score record-ID rankings."""
    ranked_cases = []
    for case in cases:
        documents = retriever.invoke(case["question"])
        retrieved_ids = []
        seen = set()
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            identifier = metadata.get("record_id") or metadata.get("doi") or metadata.get("chunk_id")
            if identifier is not None and str(identifier) not in seen:
                seen.add(str(identifier))
                retrieved_ids.append(str(identifier))
            if len(retrieved_ids) >= k:
                break
        ranked_cases.append({**case, "retrieved_ids": retrieved_ids})
    return evaluate_rankings(ranked_cases, k=k)
