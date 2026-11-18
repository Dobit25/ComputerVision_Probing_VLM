"""Audit saved results without importing ML libraries or running inference."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from .io_utils import inside_repo, sha256
from .configuration import validate_config


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "completed", "Run did not complete.")
    schema = manifest.get("artifact_schema_version", 1)
    require(type(schema) is int and schema in (1, 2), "Unsupported artifact schema.")
    expected_files = {"config.json", "environment.json", "classes.json", "predictions.csv",
                      "metrics.json", "per_class.csv", "confusion_matrix.csv"}
    if schema == 2:
        expected_files.add("samples.json")
    require(set(manifest["artifact_sha256"]) == expected_files, "Missing or unexpected artifact hashes.")
    for name, expected in manifest["artifact_sha256"].items():
        require(sha256(directory / name) == expected, f"Artifact changed: {name}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    classes = json.loads((directory / "classes.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    protocol = validate_config(config, effective=True)
    if config.get("protocol_version") == 3:
        require("mean_per_class_accuracy" in metrics, "Protocol v3 requires macro accuracy.")
    class_count = protocol["num_classes"]
    total_count = protocol["num_samples"]
    require(len(classes) == class_count, "Class count differs from dataset protocol.")
    require(len({c["dataset_name"] for c in classes}) == class_count, "Duplicate class names.")
    data = manifest["dataset"]
    require(data["name"] == config["dataset"] and data["split"] == config["split"], "Dataset identity mismatch.")
    require(data["total_test_samples"] == total_count, "Split size differs from dataset protocol.")
    require(data["download_url"] == config["dataset_download_url"], "Dataset URL mismatch.")
    require(data["torchvision_integrity_checked"] is True, "Missing dataset integrity check.")
    if config.get("protocol_version") == 3 and config["dataset"] in ("OxfordIIITPet", "DTD", "Food101"):
        require(data["test_images_archive_verified"] is True, "Missing extracted image integrity check.")
    if "partition" in protocol:
        require(data.get("partition") == protocol["partition"], "Dataset partition mismatch.")
    require(manifest["checkpoint"]["sha256"] == config["checkpoint_sha256"], "Checkpoint mismatch.")
    require(manifest["checkpoint"]["url"] == config["checkpoint_url"], "Checkpoint URL mismatch.")
    require(manifest["checkpoint"]["quick_gelu"] is True, "OpenAI checkpoint requires QuickGELU.")
    dims = manifest["embedding_dimensions"]
    require(type(dims["text"]) is int and dims["text"] > 0 and dims["image"] == dims["text"], "Invalid embedding dimensions.")
    require(dims["text_classes"] == class_count, "Text embedding class count mismatch.")
    if schema == 2:
        require(manifest["model_embedding_dim"] == dims["text"], "Model and measured embedding dimensions differ.")
        require(data["num_classes"] == class_count, "Dataset class count mismatch.")
        samples = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
    else:
        require(config["protocol_version"] == 1, "Legacy artifacts require protocol v1.")
        require(dims["text"] == 512, "Legacy OpenAI CLIP embedding dimension mismatch.")

    for i, item in enumerate(classes):
        require(item["class_id"] == i, "Class mapping is out of order.")
        prompt_name = item["dataset_name"].replace("_", " ")
        require(item["prompt_name"] == prompt_name, "Incorrect prompt class name.")
        require(item["prompt"] == config["prompt_template"].format(**{"class": prompt_name}), "Incorrect prompt.")
    with (directory / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    count = config["limit"] if config["limit"] is not None else total_count
    require(len(rows) == count == metrics["num_samples"], "Prediction count mismatch.")
    require(manifest["dataset"]["evaluated_samples"] == count, "Manifest count mismatch.")
    require(manifest["mode"] == ("full_test" if count == total_count else "smoke_test"), "Incorrect run scope.")
    require(metrics["num_classes"] == class_count, "Incorrect metric class count.")
    if schema == 2:
        require(len(samples) == count, "Input sample count mismatch.")
        require(len({s["sample_id"] for s in samples}) == count, "Duplicate input sample IDs.")
    support, correct_by_class = Counter(), Counter()
    confusion = [[0] * class_count for _ in range(class_count)]
    for index, row in enumerate(rows):
        require(int(row["test_index"]) == index, "Test indices missing, duplicated or out of order.")
        if "sample_id_template" in protocol:
            require(row["sample_id"] == protocol["sample_id_template"].format(index=index), "Sample ID mismatch.")
        if "sample_id_pattern" in protocol:
            require(re.fullmatch(protocol["sample_id_pattern"], row["sample_id"]) is not None, "Sample ID format mismatch.")
        require(row["sample_id"].startswith(protocol["namespace"] + "/"), "Sample namespace mismatch.")
        true_id, pred_id = int(row["target_id"]), int(row["prediction_id"])
        require(0 <= true_id < class_count and 0 <= pred_id < class_count, "Class ID out of bounds.")
        if schema == 2:
            require(samples[index] == {"test_index": index, "sample_id": row["sample_id"], "target_id": true_id},
                    "Prediction differs from the input sample index/ID/label.")
        require(row["target_name"] == classes[true_id]["dataset_name"], "Target name mismatch.")
        require(row["prediction_name"] == classes[pred_id]["dataset_name"], "Prediction name mismatch.")
        correct = int(true_id == pred_id)
        require(int(row["correct"]) == correct, "Incorrect correct/incorrect flag.")
        best, true = float(row["predicted_cosine"]), float(row["target_cosine"])
        margin = float(row["top1_top2_margin"])
        require(all(math.isfinite(x) for x in [best, true, margin]), "Non-finite scores.")
        require(-1.00001 <= true <= best + 1e-6 and -1.00001 <= best <= 1.00001, "Invalid cosine scores.")
        require(0 <= margin <= 2.00001, "Invalid cosine margin.")
        if correct:
            require(math.isclose(best, true, abs_tol=1e-6), "Correct prediction has inconsistent scores.")
        support[true_id] += 1
        correct_by_class[true_id] += correct
        confusion[true_id][pred_id] += 1
    correct = sum(correct_by_class.values())
    if "mean_per_class_accuracy" in metrics:
        present = [i for i in range(class_count) if support[i]]
        macro = sum(correct_by_class[i] / support[i] for i in present) / len(present)
        require(metrics["classes_with_support"] == len(present), "Present class count mismatch.")
        require(math.isclose(metrics["mean_per_class_accuracy"], macro, abs_tol=1e-12), "Macro accuracy mismatch.")
        require(math.isclose(metrics["mean_per_class_accuracy_percent"], 100 * macro, abs_tol=1e-10), "Macro percent mismatch.")
    require(correct == metrics["correct"], "Correct count differs from predictions.")
    require(math.isclose(correct / count, metrics["top1_accuracy"], abs_tol=1e-12), "Accuracy mismatch.")
    require(math.isclose(100 * correct / count, metrics["top1_accuracy_percent"], abs_tol=1e-10), "Percent mismatch.")
    with (directory / "per_class.csv").open(encoding="utf-8", newline="") as handle:
        per_class = list(csv.DictReader(handle))
    require(len(per_class) == class_count, "Per-class table is incomplete.")
    for i, row in enumerate(per_class):
        require(int(row["class_id"]) == i and row["class_name"] == classes[i]["dataset_name"], "Per-class mapping mismatch.")
        require(int(row["support"]) == support[i] and int(row["correct"]) == correct_by_class[i], "Per-class counts mismatch.")
        if support[i]:
            require(math.isclose(float(row["accuracy"]), correct_by_class[i] / support[i], abs_tol=1e-12), "Per-class accuracy mismatch.")
        else:
            require(row["accuracy"] == "", "Absent classes must have undefined accuracy.")
    with (directory / "confusion_matrix.csv").open(encoding="utf-8", newline="") as handle:
        matrix_rows = list(csv.reader(handle))
    require(matrix_rows[0] == ["true_class_id", "true_class_name", *map(str, range(class_count))], "Confusion columns mismatch.")
    require(len(matrix_rows) == class_count + 1, "Confusion rows mismatch.")
    for i, row in enumerate(matrix_rows[1:]):
        require(row[:2] == [str(i), classes[i]["dataset_name"]], "Confusion row label mismatch.")
        require(list(map(int, row[2:])) == confusion[i], "Confusion counts mismatch.")
    if schema == 2:
        full_support = data["full_split_support"]
        require(len(full_support) == class_count and all(type(n) is int and n > 0 for n in full_support)
                and sum(full_support) == total_count, "Invalid full split class support.")
        require(all(support[i] <= full_support[i] for i in range(class_count)), "Support exceeds dataset split.")
        if count == total_count:
            require([support[i] for i in range(class_count)] == full_support, "Full split class support mismatch.")
    if count == total_count and protocol.get("samples_per_class") is not None:
        require(all(support[i] == protocol["samples_per_class"] for i in range(class_count)), "Full test class balance mismatch.")
    return {"verified": True, "dataset": config["dataset"], "num_classes": class_count, "mode": manifest["mode"], "num_samples": count,
            "correct": correct, "top1_accuracy_percent": 100 * correct / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(inside_repo(root, args.run_directory)), indent=2))


if __name__ == "__main__":
    main()
