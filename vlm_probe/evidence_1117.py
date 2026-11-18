"""T0 evidence matrix (1117 plan): summaries recomputed from verified saved predictions.

No model is loaded and no inference is run. Summaries aggregate the saved
prediction fields; they do not re-derive predictions themselves. Before
aggregating, the inputs are checked at the level each artifact allows:

* probing (counting/negation): ``prediction`` is re-derived as argmax of the
  saved raw ``scores`` (lowest index on exact ties, as ``torch.argmax``), and
  ``correct`` / ``absolute_count_error`` / ``number`` are checked against it.
* Oxford-IIIT Pet: the CSV stores only predicted and target cosines, not the
  full 37-way score vector, so argmax cannot be re-derived here. Checked:
  class id/name against the run's ``classes.json``, ``correct`` against ids,
  cosine consistency, one species per class (file-name convention) and that
  convention against an independent pinned list of the 12 official cat breeds.

Outputs go to a new versioned directory and are never overwritten.

    python -m vlm_probe.evidence_1117 run     # create results/evidence_matrix_1117_v1
    python -m vlm_probe.evidence_1117 audit   # verify inputs, recompute and compare
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
PET_CLASSES = PET.parent / "classes.json"
OUT = ROOT / "results/evidence_matrix_1117_v1"
SEED = 1402
DRAWS = 2000

# Oxford-IIIT Pet (Parkhi et al., 2012): 12 cat breeds, 25 dog breeds. Pinned
# from the dataset documentation, independent of file names and model scores.
PET_CAT_BREEDS = frozenset({
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
})
PET_N_CLASSES = 37


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pet_species(sample_id: str) -> str:
    """Official Oxford-IIIT Pet convention: cat breed files start uppercase."""
    return "cat" if Path(sample_id).name[0].isupper() else "dog"


def _fail(what: str, bad: list[str]) -> None:
    if bad:
        raise ValueError(f"{what}: {len(bad)} row(s), first: {bad[:5]}")


def _argmax_lowest(scores: list[float]) -> int:
    best = 0
    for i, v in enumerate(scores):
        if v > scores[best]:
            best = i
    return best


def load_probing_rows() -> list[dict]:
    return [json.loads(line) for line in PROBING.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_probing_rows(rows: list[dict]) -> dict:
    """Re-derive counting/negation predictions from raw scores; raise on any mismatch."""
    checked = {"counting": 0, "negation": 0}
    bad_pred, bad_correct, bad_err, bad_number = [], [], [], []
    for r in rows:
        if r["task"] not in checked:
            continue
        checked[r["task"]] += 1
        sid = r["sample_id"]
        pred = _argmax_lowest(r["scores"])
        if pred != r["prediction"]:
            bad_pred.append(sid)
        if bool(r["correct"]) != (pred == r["target"]):
            bad_correct.append(sid)
        if r["task"] == "counting":
            if r["absolute_count_error"] != abs(pred - r["target"]):
                bad_err.append(sid)
            if r["number"] != r["target"] + 1:
                bad_number.append(sid)
    _fail("prediction != argmax(scores)", bad_pred)
    _fail("correct != (argmax == target)", bad_correct)
    _fail("absolute_count_error != |argmax - target|", bad_err)
    _fail("number != target + 1", bad_number)
    if not all(checked.values()):
        raise ValueError(f"missing probing tasks: {checked}")
    return {"level": "argmax re-derived from raw scores", "rows": checked}


def load_pet_rows() -> list[dict]:
    with PET.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pet_class_species(rows: list[dict], class_names: list[str] | None = None) -> dict[str, str]:
    """Map each class name to one species; raise on conflicts or unmapped predictions.

    The file-name rule is applied per image and every image of a class must agree,
    so a bad input cannot be silently overwritten by the last row of a class.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        seen[r["target_name"]].add(pet_species(r["sample_id"]))
    _fail("class maps to more than one species",
          sorted(f"{k}: {sorted(v)}" for k, v in seen.items() if len(v) != 1))
    mapping = {k: next(iter(v)) for k, v in seen.items()}
    names = set(class_names) if class_names is not None else set(mapping)
    _fail("class without test images", sorted(names - set(mapping)))
    _fail("target class not in class list", sorted(set(mapping) - names))
    _fail("prediction_name not in species mapping",
          sorted({r["prediction_name"] for r in rows} - set(mapping)))
    _fail("file-name species rule disagrees with pinned cat-breed list",
          sorted(k for k, s in mapping.items() if (s == "cat") != (k in PET_CAT_BREEDS)))
    return mapping


def verify_pet_rows(rows: list[dict], classes: list[dict]) -> dict:
    """Check Pet predictions at the level the CSV allows (no full score vector stored)."""
    id_to_name = {c["class_id"]: c["dataset_name"] for c in classes}
    if len(id_to_name) != PET_N_CLASSES or len(set(id_to_name.values())) != PET_N_CLASSES:
        raise ValueError(f"expected {PET_N_CLASSES} unique Pet classes, got {len(id_to_name)}")
    bad_ids, bad_correct, bad_cos = [], [], []
    for r in rows:
        sid, t, p = r["sample_id"], int(r["target_id"]), int(r["prediction_id"])
        if id_to_name.get(t) != r["target_name"] or id_to_name.get(p) != r["prediction_name"]:
            bad_ids.append(sid)
        if r["correct"] not in ("0", "1") or (r["correct"] == "1") != (t == p):
            bad_correct.append(sid)
        pc, tc = float(r["predicted_cosine"]), float(r["target_cosine"])
        if pc < tc or (t == p and pc != tc) or float(r["top1_top2_margin"]) < 0:
            bad_cos.append(sid)
    _fail("class id/name differs from classes.json", bad_ids)
    _fail("correct != (target_id == prediction_id)", bad_correct)
    _fail("cosine fields inconsistent with top-1 prediction", bad_cos)
    mapping = pet_class_species(rows, list(id_to_name.values()))
    return {
        "level": "consistency only; full score vector not stored, argmax not re-derived",
        "rows": len(rows),
        "classes": {s: sum(v == s for v in mapping.values()) for s in ("cat", "dog")},
        "classes_json_sha256": sha256(PET_CLASSES),
    }


def verify_inputs() -> dict:
    return {
        "probing": verify_probing_rows(load_probing_rows()),
        "pet": verify_pet_rows(load_pet_rows(), json.loads(PET_CLASSES.read_text(encoding="utf-8"))),
    }


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
    rows = load_probing_rows()
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
    rows = load_probing_rows()
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
    rows = load_pet_rows()
    class_species = pet_class_species(rows)
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
    try:
        verified = verify_inputs()
    except ValueError as exc:
        print(f"Input verification FAILED: {exc}", file=sys.stderr)
        return 1
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
    p, q = verified["probing"], verified["pet"]
    print(f"Inputs verified. Probing ({p['level']}): {p['rows']}. "
          f"Pet ({q['level']}): {q['rows']} rows, classes {q['classes']}.")
    print(f"Audit passed: {len(files)} files match recomputation from verified saved predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
