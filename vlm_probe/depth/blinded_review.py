"""Blinded independent human review sheet export and consensus scoring (P1.2).

Generates outcome-blind CSV annotation sheets (with deterministic shuffling and
separate answer keys) for the 100 confirmation COCO sources and 200 exploratory
COCO sources, and computes inter-annotator agreement (Cohen's kappa) plus
filtered 2x2 assignment metrics on the clean consensus subset.
"""

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path

from ..io_utils import sha256, write_json
from .common import ROOT, read_json, write_csv
from .confirmation import SELECTION as CONFIRMATION_SELECTION, paired_metrics, wilson
from . import natural

BLINDED_DIR = ROOT / "data/blinded_review_v1"
DEFAULT_SHUFFLE_SEED = 202611

RELATION_LABELS = (
    "to the left of",
    "to the right of",
    "above",
    "below",
)


def _load_candidates(dataset_key):
    if dataset_key == "confirmation_100":
        path = CONFIRMATION_SELECTION / "selected.json"
    elif dataset_key == "exploratory_200":
        path = ROOT / "results/depth_natural_selection_v1/selected.json"
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")
    if not path.exists():
        raise FileNotFoundError(f"Frozen candidate file not found: {path}")
    return read_json(path), path


def export_blinded_sheets(seed=DEFAULT_SHUFFLE_SEED, output_dir=None):
    """Export blinded CSV sheets and separate answer keys without CLIP scores or prior labels."""
    out_dir = Path(output_dir) if output_dir is not None else BLINDED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for dataset_key, prefix in (
        ("confirmation_100", "confirmation_100"),
        ("exploratory_200", "exploratory_200"),
    ):
        rows, src_path = _load_candidates(dataset_key)
        indexed = list(enumerate(rows))
        rng = random.Random(f"{seed}/{dataset_key}")
        rng.shuffle(indexed)

        sheet_rows = []
        answer_key = []
        for blind_idx, (orig_pos, row) in enumerate(indexed):
            blind_id = f"{prefix.upper()}_{blind_idx:03d}"
            rel_idx = int(row["target"])
            rel_text = RELATION_LABELS[rel_idx]
            prompt_text = f"a photo showing a {row['category_a']} {rel_text} a {row['category_b']}."
            sheet_rows.append({
                "blind_id": blind_id,
                "image_path": row["image_path"],
                "object_a": row["category_a"],
                "object_b": row["category_b"],
                "expected_relation": rel_text,
                "caption_to_verify": prompt_text,
                "annotator_id": "",
                "both_objects_visible": "",
                "class_names_correct": "",
                "relation_clear": "",
                "overall_accept": "",
                "exclusion_reason": "",
            })
            answer_key.append({
                "blind_id": blind_id,
                "original_position": orig_pos,
                "review_index": row["review_index"],
                "source_image_id": row["source_image_id"],
                "group_id": f"coco/{row['source_image_id']}",
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
                "target": rel_idx,
                "category_a": row["category_a"],
                "category_b": row["category_b"],
            })

        sheet_path = out_dir / f"blinded_sheet_{prefix}.csv"
        key_path = out_dir / f"answer_key_{prefix}.json"
        write_csv(sheet_path, sheet_rows)
        write_json(key_path, {
            "dataset_key": dataset_key,
            "shuffle_seed": seed,
            "source_file": src_path.relative_to(ROOT).as_posix(),
            "source_sha256": sha256(src_path),
            "num_items": len(answer_key),
            "sheet_sha256": sha256(sheet_path),
            "items": answer_key,
        })
        summary[dataset_key] = {
            "num_items": len(answer_key),
            "sheet_csv": sheet_path.relative_to(ROOT).as_posix() if out_dir == BLINDED_DIR else str(sheet_path),
            "answer_key_json": key_path.relative_to(ROOT).as_posix() if out_dir == BLINDED_DIR else str(key_path),
            "sheet_sha256": sha256(sheet_path),
        }

    return summary


def cohens_kappa(decisions_a, decisions_b):
    """Compute binary Cohen's kappa and raw agreement rate between two boolean sequences."""
    if len(decisions_a) != len(decisions_b) or not decisions_a:
        raise ValueError("Decision lists must be non-empty and have equal length.")
    n = len(decisions_a)
    a = [bool(x) for x in decisions_a]
    b = [bool(x) for x in decisions_b]
    agree = sum(x == y for x, y in zip(a, b))
    p_o = agree / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    p_e = p_a1 * p_b1 + (1.0 - p_a1) * (1.0 - p_b1)
    if abs(1.0 - p_e) <= 1e-12:
        kappa = 1.0 if abs(1.0 - p_o) <= 1e-12 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)
    return {
        "n": n,
        "raw_agreement_count": agree,
        "raw_agreement_percent": 100.0 * p_o,
        "expected_agreement_percent": 100.0 * p_e,
        "cohens_kappa": float(kappa),
        "positive_rate_a_percent": 100.0 * p_a1,
        "positive_rate_b_percent": 100.0 * p_b1,
    }


def _parse_bool_cell(value):
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "accept", "accepted"):
        return True
    if text in ("0", "false", "no", "n", "reject", "rejected"):
        return False
    raise ValueError(f"Unrecognized boolean annotation cell: {value!r}")


def load_completed_sheet(csv_path, key_path):
    """Map a completed blinded CSV sheet back to source_image_id using the answer key."""
    key_data = read_json(key_path)
    key_by_blind = {item["blind_id"]: item for item in key_data["items"]}
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(key_by_blind):
        raise ValueError(f"Row count mismatch: sheet has {len(rows)}, key has {len(key_by_blind)}.")
    mapped = {}
    for row in rows:
        bid = row["blind_id"]
        if bid not in key_by_blind:
            raise ValueError(f"Unknown blind_id in sheet: {bid}")
        item = key_by_blind[bid]
        if str(row["overall_accept"]).strip() != "":
            accepted = _parse_bool_cell(row["overall_accept"])
        else:
            accepted = (
                _parse_bool_cell(row["both_objects_visible"])
                and _parse_bool_cell(row["class_names_correct"])
                and _parse_bool_cell(row["relation_clear"])
            )
        mapped[item["source_image_id"]] = {
            "blind_id": bid,
            "source_image_id": item["source_image_id"],
            "group_id": item["group_id"],
            "accepted": accepted,
            "exclusion_reason": row.get("exclusion_reason", "").strip(),
        }
    ordered = [mapped[item["source_image_id"]] for item in sorted(key_data["items"], key=lambda x: x["original_position"])]
    return ordered


def classify_decision_branch(n_clean, p_greater_0_5, ci_low, ci_high):
    """Apply the pre-registered P1.2 decision tree (Branches A/B/C/D)."""
    if n_clean < 80:
        return {
            "branch": "D",
            "label": "underpowered_high_rejection_rate",
            "interpretation": (
                f"Clean retained sample size N_clean = {n_clean} < 80 (>20% rejected). "
                "Report bounding-box heuristic noise and statistical underpowering."
            ),
        }
    if p_greater_0_5 >= 0.05 and ci_high < 65.0:
        return {
            "branch": "A",
            "label": "null_not_significantly_above_chance",
            "interpretation": (
                f"Retained N_clean = {n_clean}; one-sided p = {p_greater_0_5:.4f} >= 0.05 and "
                f"95% CI [{ci_low:.2f}%, {ci_high:.2f}%] stays below 65%. "
                "Do not claim above-chance natural horizontal sensitivity; do not claim equivalence to 50%."
            ),
        }
    if p_greater_0_5 < 0.05 and ci_low > 50.0:
        return {
            "branch": "B",
            "label": "weak_signal_dominated_by_prior",
            "interpretation": (
                f"Retained N_clean = {n_clean}; one-sided p = {p_greater_0_5:.4f} < 0.05 and "
                f"95% CI [{ci_low:.2f}%, {ci_high:.2f}%] > 50%. "
                "Weak directional signal exists after removing prior bias in 2x2 assignment, "
                "but remains practically weak on single-image classification."
            ),
        }
    return {
        "branch": "C",
        "label": "inconclusive_wide_interval",
        "interpretation": (
            f"Retained N_clean = {n_clean}; one-sided p = {p_greater_0_5:.4f}, "
            f"95% CI [{ci_low:.2f}%, {ci_high:.2f}%]. "
            "Inconclusive due to interval width; report exact bounds."
        ),
    }


def score_confirmation_consensus(sheet_a_path, key_path=None, sheet_b_path=None):
    """Score 1 or 2 completed blinded sheets on the frozen confirmation_100 run."""
    import numpy as np

    key_file = Path(key_path) if key_path else (BLINDED_DIR / "answer_key_confirmation_100.json")
    decisions_a = load_completed_sheet(sheet_a_path, key_file)
    if sheet_b_path is not None:
        decisions_b = load_completed_sheet(sheet_b_path, key_file)
        agreement = cohens_kappa(
            [r["accepted"] for r in decisions_a],
            [r["accepted"] for r in decisions_b],
        )
        keep_ids = {
            a["group_id"]
            for a, b in zip(decisions_a, decisions_b)
            if a["accepted"] and b["accepted"]
        }
    else:
        agreement = None
        keep_ids = {a["group_id"] for a in decisions_a if a["accepted"]}

    run_dir = ROOT / "results/full_natural_confirmation_v1"
    trials = read_json(run_dir / "trials.json")
    cfg = read_json(run_dir / "config.json")
    with np.load(run_dir / "scores.npz", allow_pickle=False) as archive:
        scores = archive["scores"]

    metrics, _ = paired_metrics(trials, scores, cfg, keep_ids=keep_ids)
    primary = metrics["t2_o0"]
    branch = classify_decision_branch(
        n_clean=primary["n_sources"],
        p_greater_0_5=primary["binomial_p_greater_0_5"],
        ci_low=primary["wilson_ci95_percent"][0],
        ci_high=primary["wilson_ci95_percent"][1],
    )
    return {
        "n_total": len(decisions_a),
        "n_retained_consensus": len(keep_ids),
        "retention_percent": 100.0 * len(keep_ids) / len(decisions_a),
        "inter_annotator_agreement": agreement,
        "primary_t2_o0": primary,
        "all_conditions": metrics,
        "decision_branch": branch,
    }


def main():
    parser = argparse.ArgumentParser(description="Blinded independent review helper (P1.2).")
    sub = parser.add_subparsers(dest="action", required=True)

    p_exp = sub.add_parser("export", help="Export blinded CSV sheets and separate answer keys.")
    p_exp.add_argument("--seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    p_exp.add_argument("--output-dir", default=None)

    p_score = sub.add_parser("score", help="Score completed blinded CSV sheet(s) on confirmation_100.")
    p_score.add_argument("--sheet-a", required=True)
    p_score.add_argument("--sheet-b", default=None)
    p_score.add_argument("--key", default=None)
    p_score.add_argument("--output", default=None)

    args = parser.parse_args()
    if args.action == "export":
        res = export_blinded_sheets(seed=args.seed, output_dir=args.output_dir)
        print(res, flush=True)
    elif args.action == "score":
        res = score_confirmation_consensus(args.sheet_a, args.key, args.sheet_b)
        if args.output:
            write_json(Path(args.output), res)
        print(res, flush=True)


if __name__ == "__main__":
    main()

