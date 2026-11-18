"""Compare every prediction of two verified runs under the same protocol."""

import argparse
import csv
import json
from pathlib import Path

from .io_utils import inside_repo, sha256, write_json
from .verify import require, verify


def compare(before, after):
    audits = [verify(path) for path in (before, after)]

    def read(path, name):
        return json.loads((path / name).read_text(encoding="utf-8"))

    old_config, new_config = (read(p, "config.json") for p in (before, after))
    require(old_config == new_config, "Effective configs differ; this is not a strict refactor comparison.")
    require(read(before, "classes.json") == read(after, "classes.json"), "Class mapping or prompts differ.")
    old_manifest, new_manifest = (read(p, "manifest.json") for p in (before, after))
    for key in ("checkpoint", "preprocessing", "pretrained_config", "tokenizer", "context_length", "embedding_dimensions"):
        require(old_manifest[key] == new_manifest[key], f"Model protocol differs: {key}")
    for key, value in old_manifest["dataset"].items():
        require(new_manifest["dataset"].get(key) == value, f"Dataset provenance differs: {key}")
    old_env, new_env = (read(p, "environment.json") for p in (before, after))
    environment_differences = {k: [v, new_env.get(k)] for k, v in old_env.items() if new_env.get(k) != v}
    require(not environment_differences, f"Environment differs: {list(environment_differences)}")
    tables = []
    for path in (before, after):
        with (path / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            tables.append(list(csv.DictReader(stream)))
    require(len(tables[0]) == len(tables[1]), "Prediction counts differ.")
    changed = []
    score_fields = ("predicted_cosine", "target_cosine", "top1_top2_margin")
    max_diff = dict.fromkeys(score_fields, 0.0)
    for old, new in zip(*tables):
        for key in ("sample_id", "test_index", "target_id", "target_name"):
            require(old[key] == new[key], f"Input identity differs: {old['sample_id']} / {key}")
        if old["prediction_id"] != new["prediction_id"]:
            changed.append(old["sample_id"])
        for key in score_fields:
            max_diff[key] = max(max_diff[key], abs(float(old[key]) - float(new[key])))
    return {"equivalent_predictions": not changed, "samples_compared": len(tables[0]),
            "changed_prediction_count": len(changed), "changed_sample_ids": changed,
            "max_absolute_score_difference": max_diff,
            "prediction_csv_byte_identical": sha256(before / "predictions.csv") == sha256(after / "predictions.csv"),
            "same_effective_config": True, "same_environment": True, "audits": audits,
            "runs": [{"name": p.name, "manifest_sha256": sha256(p / "manifest.json"),
                      "predictions_sha256": sha256(p / "predictions.csv")} for p in (before, after)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--output", help="Optional new comparison JSON inside the repository.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = inside_repo(root, args.output) if args.output else None
    if output is not None and output.exists():
        raise FileExistsError(output)
    result = compare(inside_repo(root, args.before), inside_repo(root, args.after))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    print(json.dumps(result, indent=2))
    if not result["equivalent_predictions"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
