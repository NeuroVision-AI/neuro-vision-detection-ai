from __future__ import annotations

import unittest

import numpy as np

from src.xai_evaluation import evaluate_explanation, localization_iou, pointing_game, rank_correlation


class XAIEvaluationTests(unittest.TestCase):
    def test_repeatability_and_localization(self):
        heatmap = np.arange(16, dtype=float).reshape(4, 4)
        mask = np.zeros((4, 4), dtype=int)
        mask[2:, 2:] = 1
        self.assertAlmostEqual(rank_correlation(heatmap, heatmap.copy()), 1.0)
        self.assertEqual(pointing_game(heatmap, mask), 1.0)
        self.assertGreater(localization_iou(heatmap, mask, quantile=0.75), 0.0)

    def test_randomization_sensitivity_is_reported(self):
        heatmap = np.arange(9, dtype=float).reshape(3, 3)
        randomized = np.flipud(heatmap)
        result = evaluate_explanation(heatmap, randomized=randomized)
        self.assertIn("randomization_sensitivity", result)
        self.assertGreaterEqual(result["randomization_sensitivity"], 0.0)


if __name__ == "__main__":
    unittest.main()
