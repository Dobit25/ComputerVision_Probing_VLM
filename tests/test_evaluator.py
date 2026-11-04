"""Exercise the real evaluator with small in-memory dual-encoder/dataset adapters."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_probe.dataset_adapters import DatasetBundle
from vlm_probe.evaluate import evaluate
from vlm_probe.io_utils import configure_paths

ROOT = Path(__file__).resolve().parents[1]


class EvaluatorTests(unittest.TestCase):
    def test_dynamic_dimensions_and_final_partial_batch(self):
        configure_paths(ROOT)
        import torch

        class FixtureModel:
            embedding_dim = 7
            preprocess = None
            metadata = {"model_embedding_dim": 7}

            def encode_texts(self, texts):
                if len(texts) != 3:
                    raise AssertionError("Expected all three candidate prompts.")
                return torch.eye(3, 7)

            def encode_images(self, images):
                return images

        labels = [0, 0, 0, 1, 1, 2]
        predictions = [0, 1, 0, 1, 2, 2]
        dataset = [(torch.eye(3, 7)[p], t) for t, p in zip(labels, predictions)]
        bundle = DatasetBundle(dataset, labels, ["a", "b", "c"],
                               [f"fixture/test/{i}" for i in range(6)], {"name": "fixture", "split": "test"}).validate()
        config = {"device": "cpu", "seed": 0, "cpu_threads": 1,
                  "prompt_template": "a photo of a {class}.", "limit": None, "batch_size": 5}
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as tmp:
            output = Path(tmp)
            manifest = {}
            with patch("vlm_probe.evaluate.load_model", return_value=FixtureModel()), \
                    patch("vlm_probe.evaluate.load_dataset", return_value=bundle), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluate(ROOT, output, config, manifest)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual((metrics["num_samples"], metrics["num_classes"], metrics["correct"]), (6, 3, 4))
            self.assertEqual(manifest["embedding_dimensions"], {"image": 7, "text": 7, "text_classes": 3})
            self.assertEqual(len((output / "confusion_matrix.csv").read_text().splitlines()), 4)
