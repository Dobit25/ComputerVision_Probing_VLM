import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_probe.configuration import validate_config
from vlm_probe.dataset_adapters import DatasetBundle
from vlm_probe.io_utils import sha256, write_json
from vlm_probe.metrics import classification_metrics
from vlm_probe.verify import verify

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_legacy_baselines_remain_readable(self):
        for name, correct in (("smoke_cifar100_clip_b32_v1", 139), ("full_cifar100_clip_b32_v1", 6448)):
            self.assertEqual(verify(ROOT / "results" / name)["correct"], correct)

    def test_invalid_protocol_and_limits(self):
        config = json.loads((ROOT / "results/full_cifar100_clip_b32_v1/config.json").read_text())
        for key, value in (("dataset", "unknown"), ("limit", 10001), ("limit", True),
                           ("selection", "shuffled"), ("checkpoint_sha256", "incorrect"),
                           ("prompt_template", "a drawing of a {class}.")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**config, key: value}, effective=True)

    def test_bundle_rejects_duplicate_ids_and_invalid_labels(self):
        for targets, ids in (([0, 1], ["a", "a"]), ([0, 2], ["a", "b"]), ([0], ["a", "b"])):
            with self.subTest(targets=targets, ids=ids), self.assertRaises(ValueError):
                DatasetBundle([0, 1], targets, ["a", "b"], ids, {}).validate()


class GenericVerifierTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT / ".cache" / "tests"
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        # Deliberately unlike CIFAR: 3 classes, unequal support, 7-dimensional vectors.
        self.protocol = {"num_classes": 3, "num_samples": 6, "namespace": "fixture/test"}
        self.mock_protocol = patch("vlm_probe.verify.validate_config", return_value=self.protocol)
        self.mock_protocol.start()
        self.addCleanup(self.mock_protocol.stop)
        self.make_run()

    def write_csv(self, name, fields, rows):
        with (self.path / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def make_run(self, limit=None):
        targets, predictions = [0, 0, 0, 1, 1, 2], [0, 1, 0, 1, 2, 2]
        if limit is not None:
            targets, predictions = targets[:limit], predictions[:limit]
        classes = [{"class_id": i, "dataset_name": n, "prompt_name": n, "prompt": f"a photo of a {n}."}
                   for i, n in enumerate(["a", "b", "c"])]
        config = {"dataset": "fixture", "split": "test", "dataset_download_url": "source",
                  "checkpoint_sha256": "hash", "checkpoint_url": "weights",
                  "limit": limit, "prompt_template": "a photo of a {class}."}
        samples = [{"test_index": i, "sample_id": f"fixture/test/{i}", "target_id": t} for i, t in enumerate(targets)]
        rows = [{**sample, "target_name": classes[t]["dataset_name"], "prediction_id": p,
                 "prediction_name": classes[p]["dataset_name"], "correct": int(t == p),
                 "predicted_cosine": 0.8, "target_cosine": 0.8 if t == p else 0.1, "top1_top2_margin": 0.2}
                for sample, t, p in zip(samples, targets, predictions)]
        metrics, per_class, matrix = classification_metrics(targets, predictions, ["a", "b", "c"])
        for name, payload in {"config": config, "samples": samples, "classes": classes,
                              "metrics": metrics, "environment": {}}.items():
            write_json(self.path / f"{name}.json", payload)
        self.write_csv("predictions.csv", list(rows[0]), rows)
        self.write_csv("per_class.csv", list(per_class[0]), per_class)
        with (self.path / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true_class_id", "true_class_name", 0, 1, 2])
            for i, row in enumerate(matrix):
                writer.writerow([i, classes[i]["dataset_name"], *row])
        self.manifest = {"status": "completed", "artifact_schema_version": 2,
                         "mode": "full_test" if limit is None else "smoke_test",
                         "checkpoint": {"sha256": "hash", "url": "weights", "quick_gelu": True},
                         "model_embedding_dim": 7, "embedding_dimensions": {"image": 7, "text": 7, "text_classes": 3},
                         "dataset": {"name": "fixture", "split": "test", "download_url": "source",
                                     "torchvision_integrity_checked": True, "num_classes": 3,
                                     "total_test_samples": 6, "evaluated_samples": len(targets),
                                     "full_split_support": [3, 2, 1]}}
        self.refresh_hashes()

    def refresh_hashes(self):
        self.manifest["artifact_sha256"] = {p.name: sha256(p) for p in self.path.iterdir() if p.name != "manifest.json"}
        write_json(self.path / "manifest.json", self.manifest)

    def test_non_cifar_shapes_and_unequal_class_support(self):
        result = verify(self.path)
        self.assertEqual((result["num_classes"], result["num_samples"], result["correct"]), (3, 6, 4))

    def test_smoke_with_absent_classes(self):
        self.make_run(limit=2)
        self.assertEqual(verify(self.path)["mode"], "smoke_test")

    def test_changed_input_label_rejected_even_with_refreshed_hash(self):
        samples = json.loads((self.path / "samples.json").read_text())
        samples[0]["target_id"] = 1
        write_json(self.path / "samples.json", samples)
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "input sample"):
            verify(self.path)

    def test_missing_prediction_rejected_even_with_refreshed_hash(self):
        p = self.path / "predictions.csv"
        p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "Prediction count"):
            verify(self.path)

    def test_bad_embedding_dimensions_rejected(self):
        self.manifest["embedding_dimensions"]["image"] = 8
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "embedding dimensions"):
            verify(self.path)

    def test_artifact_hash_change_rejected(self):
        with (self.path / "predictions.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Artifact changed"):
            verify(self.path)

    def test_wrong_full_split_support_rejected(self):
        self.manifest["dataset"]["full_split_support"] = [2, 2, 2]
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "Support exceeds"):
            verify(self.path)
