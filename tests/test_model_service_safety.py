from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from api.services.model_service import ModelService, ModelUnavailableError
from src import config


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 4)

    def forward(self, value):
        return self.layer(value)


class ModelServiceSafetyTests(unittest.TestCase):
    def test_missing_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "MODEL_SAVE_DIR", Path(tmp)), patch(
            "api.services.model_service.get_model", return_value=TinyModel()
        ):
            service = ModelService()
            with self.assertRaises(ModelUnavailableError):
                service.load_model("custom_cnn")

    def test_nested_checkpoint_and_calibration_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "custom_cnn"
            model_dir.mkdir(parents=True)
            model = TinyModel()
            torch.save(
                {"model_state_dict": model.state_dict(), "class_names": list(config.CLASS_NAMES)},
                model_dir / config.CHECKPOINT_BEST_ACC,
            )
            (model_dir / "calibration.json").write_text(
                json.dumps({"temperature": 1.7}), encoding="utf-8"
            )
            with patch.object(config, "MODEL_SAVE_DIR", Path(tmp)), patch(
                "api.services.model_service.get_model", return_value=TinyModel()
            ):
                service = ModelService()
                service.load_model("custom_cnn")
                self.assertTrue(service.model_metadata_cache["custom_cnn"]["calibrated"])
                self.assertAlmostEqual(
                    service.model_metadata_cache["custom_cnn"]["temperature"], 1.7
                )


if __name__ == "__main__":
    unittest.main()
