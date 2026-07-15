from __future__ import annotations

import unittest

from scripts.audit_external_candidate import BKTree, hamming


class ExternalCandidateAuditTests(unittest.TestCase):
    def test_hamming_and_metric_tree_radius(self):
        self.assertEqual(hamming(0b0000, 0b1011), 3)
        tree = BKTree()
        for value in (0b0000, 0b1111, 0b1010):
            tree.add(value)
        matches = sorted(tree.query(0b1000, radius=1))
        self.assertEqual(matches, [(1, 0b0000), (1, 0b1010)])

    def test_duplicate_values_are_stored_once(self):
        tree = BKTree()
        tree.add(7)
        tree.add(7)
        self.assertEqual(tree.query(7, radius=0), [(0, 7)])


if __name__ == "__main__":
    unittest.main()
