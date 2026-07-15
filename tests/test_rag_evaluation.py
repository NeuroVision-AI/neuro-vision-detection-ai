from __future__ import annotations

import unittest

import pandas as pd

from api.services.rag_service import find_paper_header_row
from src.rag_evaluation import evaluate_rankings, evaluate_retriever, score_ranked_ids


class RAGEvaluationTests(unittest.TestCase):
    def test_formatted_workbook_header_is_detected(self):
        frame = pd.DataFrame(
            [
                ["Normalized Paper Index", None, None],
                [None, None, None],
                ["#", "Paper Title", "Year"],
                [1, "Example", 2025],
            ]
        )
        self.assertEqual(find_paper_header_row(frame), 2)

    def test_companion_header_is_detected(self):
        frame = pd.DataFrame(
            [["Normalized Paper Index", None], ["ID", "Paper title"], [1, "Example"]]
        )
        self.assertEqual(find_paper_header_row(frame), 1)

    def test_rank_metrics(self):
        metrics = score_ranked_ids(["9", "2", "3"], ["2", "4"], k=3)
        self.assertAlmostEqual(metrics["precision_at_k"], 1 / 3)
        self.assertAlmostEqual(metrics["recall_at_k"], 0.5)
        self.assertAlmostEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["hit_at_k"], 1.0)

    def test_aggregate_metrics(self):
        result = evaluate_rankings(
            [
                {"question": "q1", "relevant_ids": ["1"], "retrieved_ids": ["1", "2"]},
                {"question": "q2", "relevant_ids": ["4"], "retrieved_ids": ["2", "4"]},
            ],
            k=2,
        )
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["aggregate"]["hit_at_k"], 1.0)

    def test_retriever_deduplicates_chunks_by_record(self):
        class Document:
            def __init__(self, record_id):
                self.metadata = {"record_id": record_id}

        class Retriever:
            def invoke(self, _question):
                return [Document("1"), Document("1"), Document("2"), Document("3")]

        result = evaluate_retriever(
            Retriever(),
            [{"question": "q", "relevant_ids": ["2"]}],
            k=2,
        )
        self.assertEqual(result["aggregate"]["hit_at_k"], 1.0)
        self.assertEqual(result["aggregate"]["reciprocal_rank"], 0.5)


if __name__ == "__main__":
    unittest.main()
