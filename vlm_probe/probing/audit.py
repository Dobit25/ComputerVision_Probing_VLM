"""Recompute probing scores/metrics and check stimuli labels without model inference."""

import argparse
import csv
import hashlib
import json
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace

from ..assets import digest_file
from ..io_utils import inside_repo, sha256
from ..verify import require, verify as verify_classification
from .evaluate import fine_samples
from .metrics import score_row, summarize
from .stimuli import scenes, ordered_samples


def verify(root, directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["status"] == "completed" and manifest["kind"] == "core_probing_v1", "Probe run incomplete or wrong kind.")
    names = {"config.json", "environment.json", "samples.json", "predictions.jsonl", "metrics.json"}
    require(set(manifest["artifact_sha256"]) == names, "Probe artifact set mismatch.")
    for name, digest in manifest["artifact_sha256"].items():
        require(sha256(directory / name) == digest, f"Changed probe artifact: {name}")
    config = json.loads((directory / "config.json").read_text())
    model_config = json.loads((root / config["model_config"]).read_text())
    require(manifest["checkpoint"]["sha256"] == model_config["checkpoint_sha256"], "Probe checkpoint mismatch.")
    require(manifest["checkpoint"]["url"] == model_config["checkpoint_url"], "Probe checkpoint URL mismatch.")
    require(manifest["checkpoint"]["quick_gelu"] is True, "Probe activation mismatch.")
    require(manifest["embedding_dim"] == manifest["model_embedding_dim"] == 512, "Probe embedding dimension mismatch.")
    snapshot = inside_repo(root, manifest["source_snapshot"]["path"])
    require(sha256(snapshot) == manifest["source_snapshot"]["sha256"], "Source snapshot checksum mismatch.")
    with zipfile.ZipFile(snapshot) as archive:
        for name, digest in manifest["source_sha256"].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, "Source snapshot content mismatch.")
    synthetic = scenes(config["renderer_seed"])
    source = root / "data/cifar-100-python"
    require(digest_file(source / "test", "md5") == "f0ef6b0ae62326f3e7ffdfab6717acfc", "CIFAR checksum mismatch.")
    require(digest_file(source / "meta", "md5") == "7973b15100ade9c7d40fb424638fde48", "CIFAR meta checksum mismatch.")
    with (source / "test").open("rb") as stream:
        test = pickle.load(stream, encoding="latin1")
    with (source / "meta").open("rb") as stream:
        meta = pickle.load(stream, encoding="latin1")
    natural = fine_samples(root, SimpleNamespace(targets=test["fine_labels"], classes=meta["fine_label_names"],
                           sample_ids=[f"cifar100/test/{i:05d}" for i in range(len(test["fine_labels"]))]))
    require(manifest["mode"] in ("full", "smoke"), "Unknown probe run scope.")
    expected = ordered_samples(synthetic + natural, config["tasks"], 64 if manifest["mode"] == "smoke" else None)
    samples = json.loads((directory / "samples.json").read_text())
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    require(len(samples) == len(rows) == len(expected) == manifest["num_samples"], "Probe sample count mismatch.")
    unique = {task: len({s["image_sha256"] for s in samples if s["task"] == task})
              for task in ("counting", "spatial", "negation")}
    require(unique == manifest["synthetic_unique_images"], "Unique image count mismatch.")
    if manifest["mode"] == "full":
        require(unique == {"counting": 1080, "spatial": 720, "negation": 720}, "Unexpected duplicate images.")
    baseline = inside_repo(root, config["baseline_run"])
    verify_classification(baseline)
    require(sha256(baseline / "manifest.json") == manifest["baseline_manifest_sha256"], "Baseline reference changed.")
    with (baseline / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        baseline_rows = list(csv.DictReader(stream))
    checked_images = set()
    for sample, row, truth in zip(samples, rows, expected):
        core = {k: v for k, v in sample.items() if k not in ("image_path", "image_sha256")}
        require(core == truth, "Probe definition differs from the frozen generator/official CIFAR labels.")
        if "image_path" in sample:
            path = inside_repo(root, sample["image_path"])
            if sample["image_path"] not in checked_images:
                require(sha256(path) == sample["image_sha256"], "Probe image checksum mismatch.")
                checked_images.add(sample["image_path"])
        expected_row = score_row(sample, row["scores"], row.get("control_scores"), row.get("global_prediction_id"))
        require(expected_row == row, "Probe prediction/flags/margins inconsistent with scores.")
        if sample["task"] == "fine_grained":
            ref = baseline_rows[sample["test_index"]]
            require(int(ref["target_id"]) == sample["global_target_id"], "Natural ground truth mismatch.")
            require(int(ref["prediction_id"]) == row["global_prediction_id"], "Global prediction changed from baseline.")
    recomputed = summarize(rows)
    require(recomputed == json.loads((directory / "metrics.json").read_text()), "Probe summary/CI mismatch.")
    return {"verified": True, "mode": manifest["mode"], "num_samples": len(rows),
            "global_fine_grained_predictions_match_baseline": True,
            "tasks": {task: {k: value[k] for k in ("num_samples", "correct", "accuracy_percent", "accuracy_ci95_group_bootstrap")}
                      for task, value in recomputed.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(verify(root, inside_repo(root, args.run_directory)), indent=2))


if __name__ == "__main__":
    main()
