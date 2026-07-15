from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data_integrity import (
    assign_grouped_splits,
    audit_manifest,
    build_manifest,
    materialize_splits,
    prepare_research_dataset,
)


def write_pattern(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(24, 24), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


class DataIntegrityTests(unittest.TestCase):
    def test_official_test_and_duplicate_groups_never_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            write_pattern(root / "Training" / "glioma" / "patientA_slice1.png", 1)
            write_pattern(root / "Training" / "glioma" / "patientA_slice2.png", 2)
            write_pattern(root / "Training" / "meningioma" / "patientB_slice1.png", 3)
            write_pattern(root / "Testing" / "glioma" / "patientC_slice1.png", 4)

            # Exact copy of the official-test image placed in Training: the whole
            # duplicate group must be locked to test.
            duplicate = root / "Training" / "glioma" / "duplicate_of_C.png"
            duplicate.write_bytes((root / "Testing" / "glioma" / "patientC_slice1.png").read_bytes())

            records = build_manifest(root, patient_pattern=r"(?P<patient>patient[A-Z])")
            assign_grouped_splits(records, seed=7, preserve_source_test=True)
            audit = audit_manifest(records)
            self.assertTrue(audit["leakage_free"])
            official_test_hash = next(
                record.exact_sha256 for record in records if record.source_split == "test"
            )
            self.assertEqual(
                {record.assigned_split for record in records if record.exact_sha256 == official_test_hash},
                {"test"},
            )
            patient_a_splits = {
                record.assigned_split for record in records if record.patient_id == "patientA"
            }
            self.assertEqual(len(patient_a_splits), 1)

    def test_materialization_refuses_nonempty_output_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output = Path(tmp) / "processed"
            write_pattern(root / "glioma" / "a.png", 10)
            write_pattern(root / "meningioma" / "b.png", 11)
            records = build_manifest(root)
            assign_grouped_splits(records, seed=1, preserve_source_test=False)
            materialize_splits(records, root, output)
            with self.assertRaises(FileExistsError):
                materialize_splits(records, root, output)

    def test_conflicting_duplicate_group_requires_explicit_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            write_pattern(root / "Training" / "glioma" / "conflict.png", 20)
            duplicate = root / "Training" / "meningioma" / "conflict_copy.png"
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            duplicate.write_bytes((root / "Training" / "glioma" / "conflict.png").read_bytes())
            write_pattern(root / "Training" / "glioma" / "valid_glioma.png", 21)
            write_pattern(root / "Training" / "meningioma" / "valid_meningioma.png", 22)

            with self.assertRaises(ValueError):
                prepare_research_dataset(
                    root, Path(tmp) / "blocked", Path(tmp) / "blocked_manifest", "test"
                )
            self.assertTrue((Path(tmp) / "blocked_manifest" / "dataset_audit_pre_exclusion.json").exists())

            audit = prepare_research_dataset(
                root,
                Path(tmp) / "processed",
                Path(tmp) / "manifest",
                "test",
                exclude_conflicting_groups=True,
            )
            self.assertEqual(audit["excluded_conflicting_record_count"], 2)
            self.assertTrue(audit["label_consistent"])


if __name__ == "__main__":
    unittest.main()
