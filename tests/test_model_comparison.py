from __future__ import annotations

import unittest

import numpy as np

from scripts.compare_models import paired_group_bootstrap


class ModelComparisonTests(unittest.TestCase):
    def test_paired_group_bootstrap_is_deterministic_and_directional(self):
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        primary = y_true.copy()
        comparator = np.array([0, 1, 1, 1, 2, 0, 3, 3])
        groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
        first = paired_group_bootstrap(
            y_true, primary, comparator, groups, samples=100, seed=42
        )
        second = paired_group_bootstrap(
            y_true, primary, comparator, groups, samples=100, seed=42
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["accuracy_difference"]["lower_95"], 0.0)
        self.assertGreaterEqual(first["macro_f1_difference"]["lower_95"], 0.0)


if __name__ == "__main__":
    unittest.main()
