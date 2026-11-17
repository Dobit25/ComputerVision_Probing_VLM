"""T0 evidence matrix (1117 plan): recompute summaries from committed scores.

No model is loaded and no inference is run. Inputs are existing predictions;
outputs go to a new versioned directory and are never overwritten.

    python -m vlm_probe.evidence_1117 run     # create results/evidence_matrix_1117_v1
    python -m vlm_probe.evidence_1117 audit   # recompute and compare with saved outputs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBING = ROOT / "results/full_probing_clip_b32_v1/predictions.jsonl"
PET = ROOT / "results/full_oxford_pet_clip_b32_v3/predictions.csv"
OUT = ROOT / "results/evidence_matrix_1117_v1"
SEED = 1402
DRAWS = 2000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pet_species(sample_id: str) -> str:
    """Official Oxford-IIIT Pet convention: cat breed files start uppercase."""
    return "cat" if Path(sample_id).name[0].isupper() else "dog"


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap_rate(flags: list[int], seed: int = SEED, draws: int = DRAWS) -> list[float]:
    rng = random.Random(seed)
    n = len(flags)
    stats = [100.0 * sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)]
    return [percentile(stats, 0.025), percentile(stats, 0.975)]


def counting_summary() -> dict:
    rows = [json.loads(line) for line in PROBING.read_text(encoding="utf-8").splitlines() if line.strip()]
    counting = [r for r in rows if r["task"] == "counting"]
    confusion = [[0] * 6 for _ in range(6)]
    for r in counting:
        confusion[r["target"]][r["prediction"]] += 1
    groups: dict[str, dict] = {}
    for key in ("condition", "stratum", "color", "shape"):
        acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in counting:
            acc[str(r[key])][0] += int(r["correct"])
            acc[str(r[key])][1] += 1
        groups[key] = {k: {"correct": c, "n": n} for k, (c, n) in sorted(acc.items())}
    n = len(counting)
    return {
        "n": n,
        "correct": sum(int(r["correct"]) for r in counting),
        "mae": sum(r["absolute_count_error"] for r in counting) / n,
        "mean_signed_error": sum(r["prediction"] - r["target"] for r in counting) / n,
        "confusion_rows_true_cols_pred": confusion,
        "groups": groups,
    }


def negation_summary() -> dict:
    rows = [json.loads(line) for line in PROBING.read_text(encoding="utf-8").splitlines() if line.strip()]
    neg = [r for r in rows if r["task"] == "negation"]
    out: dict[str, dict] = {}
    for key in ("condition", "stratum"):
        acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in neg:
            acc[str(r[key])][0] += int(r["correct"])
            acc[str(r[key])][1] += 1
        out[key] = {k: {"correct": c, "n": n} for k, (c, n) in sorted(acc.items())}
    return {"n": len(neg), "correct": sum(int(r["correct"]) for r in neg), "groups": out}


def pet_summary() -> tuple[dict, list[dict]]:
    with PET.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    class_species = {r["target_name"]: pet_species(r["sample_id"]) for r in rows}
    errors = [r for r in rows if r["correct"] == "0"]
    cross_flags = [int(class_species[r["target_name"]] != class_species[r["prediction_name"]]) for r in errors]
    all_cross = [int(r["correct"] == "0" and class_species[r["target_name"]] != class_species[r["prediction_name"]]) for r in rows]
    per_species = []
    for s in ("cat", "dog"):
        n_rows = [r for r in rows if class_species[r["target_name"]] == s]
        errs = [r for r in n_rows if r["correct"] == "0"]
        cross = [r for r in errs if class_species[r["prediction_name"]] != s]
        per_species.append({
            "species": s,
            "n_classes": sum(1 for v in class_species.values() if v == s),
            "n_images": len(n_rows),
            "errors": len(errs),
            "cross_species_errors": len(cross),
            "within_species_errors": len(errs) - len(cross),
        })
    summary = {
        "n_images": len(rows),
        "errors": len(errors),
        "cross_species_errors": sum(cross_flags),
        "within_species_errors": len(errors) - sum(cross_flags),
        "cross_share_of_errors_percent": 100.0 * sum(cross_flags) / len(errors),
        "cross_share_of_errors_ci95_percent": bootstrap_rate(cross_flags),
        "cross_species_rate_all_images_percent": 100.0 * sum(all_cross) / len(rows),
        "cross_species_rate_all_images_ci95_percent": bootstrap_rate(all_cross),
        "bootstrap": {"unit": "image", "seed": SEED, "draws": DRAWS},
        "species_rule": "Oxford-IIIT Pet official convention: cat-breed file names start uppercase",
        "per_species": per_species,
    }
    return summary, per_species


def build() -> dict[str, str]:
    counting = counting_summary()
    negation = negation_summary()
    pet, per_species = pet_summary()
    metrics = {
        "new_inference": False,
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in (PROBING, PET)},
        "counting": counting,
        "negation": negation,
        "oxford_pet_species": pet,
    }
    files = {"metrics.json": json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"}
    lines = ["true_count," + ",".join(f"pred_{i}" for i in range(1, 7))]
    lines += [f"{i + 1}," + ",".join(map(str, row)) for i, row in enumerate(counting["confusion_rows_true_cols_pred"])]
    files["counting_confusion.csv"] = "\n".join(lines) + "\n"
    lines = ["task,group_by,value,n,correct,accuracy_percent"]
    for task, summ in (("counting", counting), ("negation", negation)):
        for key, vals in summ["groups"].items():
            for v, d in vals.items():
                lines.append(f"{task},{key},{v},{d['n']},{d['correct']},{100.0 * d['correct'] / d['n']:.4f}")
    files["probing_breakdown_recomputed.csv"] = "\n".join(lines) + "\n"
    cols = list(per_species[0])
    files["pet_species.csv"] = "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in per_species]) + "\n"
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "audit"])
    args = parser.parse_args(argv)
    files = build()
    if args.command == "run":
        if OUT.exists():
            print(f"{OUT} already exists; use audit or a new version.", file=sys.stderr)
            return 1
        OUT.mkdir(parents=True)
        for name, text in files.items():
            (OUT / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {len(files)} files to {OUT}")
        return 0
    bad = [n for n, t in files.items() if not (OUT / n).exists() or (OUT / n).read_text(encoding="utf-8") != t]
    if bad:
        print("Audit FAILED: " + ", ".join(bad), file=sys.stderr)
        return 1
    print(f"Audit passed: {len(files)} files match recomputation from scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
