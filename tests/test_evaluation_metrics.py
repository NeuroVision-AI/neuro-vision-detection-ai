from __future__ import annotations

import unittest

import numpy as np

from src.evaluate import (
    bootstrap_confidence_interval,
    compute_research_metrics,
    expected_calibration_error,
    multiclass_brier_score,
    risk_coverage_curve,
)


class EvaluationMetricTests(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        self.probabilities = np.array(
            [
                [0.90, 0.04, 0.03, 0.03],
                [0.80, 0.10, 0.05, 0.05],
                [0.05, 0.85, 0.05, 0.05],
                [0.15, 0.70, 0.10, 0.05],
                [0.05, 0.05, 0.85, 0.05],
                [0.10, 0.10, 0.70, 0.10],
                [0.05, 0.05, 0.10, 0.80],
                [0.10, 0.10, 0.10, 0.70],
            ]
        )

    def test_research_metrics_include_discrimination_calibration_and_ci(self):
        metrics = compute_research_metrics(
            self.y_true,
            self.probabilities,
            ["a", "b", "c", "d"],
            groups=np.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"]),
            bootstrap_samples=50,
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["mcc"], 1.0)
        self.assertIn("expected_calibration_error", metrics)
        self.assertIn("multiclass_brier_score", metrics)
        self.assertEqual(
            metrics["confidence_intervals"]["accuracy"]["bootstrap_unit"],
            "patient_or_duplicate_group",
        )
        self.assertEqual(metrics["per_class"]["a"]["specificity"], 1.0)

    def test_calibration_helpers_have_expected_bounds(self):
        ece = expected_calibration_error(self.y_true, self.probabilities, n_bins=5)
        brier = multiclass_brier_score(self.y_true, self.probabilities)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertGreaterEqual(brier, 0.0)
        curve = risk_coverage_curve(self.y_true, self.probabilities, points=5)
        self.assertEqual(curve[-1]["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
