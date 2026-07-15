#!/usr/bin/env python3
"""Evaluate literature retrieval without requiring an LLM answer generator."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.services.rag_service import RAGService
from src.rag_evaluation import evaluate_retriever, load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark row-level literature retrieval")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "research" / "rag_benchmark_template.jsonl",
    )
    parser.add_argument(
        "--benchmark-metadata",
        type=Path,
        default=PROJECT_ROOT / "research" / "rag_benchmark_metadata.json",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rag_evaluation" / "retrieval_metrics.json",
    )
    args = parser.parse_args()

    cases = load_benchmark(args.benchmark)
    service = RAGService()
    fetch_k = max(args.k * 6, 30)
    retriever = service.vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": fetch_k}
    )
    results = evaluate_retriever(retriever, cases, k=args.k)
    results["created_utc"] = datetime.now(timezone.utc).isoformat()
    results["benchmark"] = str(args.benchmark)
    results["benchmark_metadata"] = (
        json.loads(args.benchmark_metadata.read_text(encoding="utf-8"))
        if args.benchmark_metadata.exists()
        else {"domain_expert_reviewed": False, "status": "metadata_missing"}
    )
    results["knowledge_base"] = service.get_collection_stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["aggregate"], indent=2))
    if not results["benchmark_metadata"].get("domain_expert_reviewed"):
        print("WARNING: benchmark is not domain-expert reviewed; metrics are preliminary engineering results.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
