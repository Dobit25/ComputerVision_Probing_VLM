"""Prior-to-Signal Ratio (PSR) decomposition for 2x2 spatial score matrices (P2.1).

Proves quantitatively why CLIP's 2x2 assignment accuracy (~50-56%) coexists with
near-zero single-image pair accuracy (both_images_correct ~0.56-4.00%): the
position-invariant prior bias |B| dominates the spatial flip signal |Delta_spatial|.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..depth.common import ROOT, read_json, cluster_ci
from ..io_utils import write_json

CONFIG_PATH = ROOT / "configs/spatial_diagnostics_v1.json"


def decompose_pair_matrix(matrix):
    """Decompose a 2x2 score matrix [[s(I0,c0), s(I0,c1)], [s(I1,c0), s(I1,c1)]].

    Returns:
      - d0: s(I0, c0) - s(I0, c1)  (correct on I0 iff d0 > 0)
      - d1: s(I1, c0) - s(I1, c1)  (correct on I1 iff d1 < 0)
      - prior_bias_B: 0.5 * (d0 + d1)  (position-invariant preference for c0 over c1)
      - spatial_signal_delta: 0.5 * (d0 - d1) = 0.5 * assignment_gap
      - assignment_gap: d0 - d1
      - abs_prior_to_signal_ratio: |B| / max(|Delta_spatial|, 1e-12)
      - assignment_correct: Delta_spatial > 0
      - both_images_correct: (d0 > 0) and (d1 < 0), equivalent to Delta_spatial > |B|
      - same_label_collapse: (d0 > 0 and d1 > 0) or (d0 < 0 and d1 < 0), i.e., |B| > |Delta_spatial|
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 2) or not np.isfinite(m).all():
        raise ValueError(f"Expected finite 2x2 matrix, got shape {m.shape}.")
    d0 = float(m[0, 0] - m[0, 1])
    d1 = float(m[1, 0] - m[1, 1])
    prior_bias = 0.5 * (d0 + d1)
    spatial_signal = 0.5 * (d0 - d1)
    gap = d0 - d1
    abs_b = abs(prior_bias)
    abs_sig = abs(spatial_signal)
    psr = abs_b / max(abs_sig, 1e-12)
    decisions = m.argmax(axis=1)
    both_correct = bool(decisions[0] == 0 and decisions[1] == 1)
    same_prediction = bool(decisions[0] == decisions[1])
    return {
        "d0": d0,
        "d1": d1,
        "prior_bias_B": prior_bias,
        "abs_prior_bias": abs_b,
        "spatial_signal_delta": spatial_signal,
        "abs_spatial_signal": abs_sig,
        "assignment_gap": gap,
        "prior_to_signal_ratio": psr,
        "assignment_correct": bool(gap > 0.0),
        "both_images_correct": both_correct,
        "same_prediction_collapse": same_prediction,
        "signal_exceeds_prior": bool(spatial_signal > abs_b),
    }


def analyze_run_pairs(run_dir, bootstrap_draws=2000, bootstrap_seed=1402):
    """Decompose all horizontal/vertical 2x2 pairs in a frozen spatial or confirmation run."""
    run_path = Path(run_dir)
    manifest = read_json(run_path / "manifest.json")
    stage = manifest.get("stage", "unknown")
    trials = read_json(run_path / "trials.json")
    with np.load(run_path / "scores.npz", allow_pickle=False) as archive:
        scores = archive["scores"]

    is_natural = stage in ("natural", "natural_confirmation")
    groups = defaultdict(list)
    for idx, row in enumerate(trials):
        if is_natural and row["target"] not in (0, 1):
            continue
        axis = row["target"] // 2
        variant = "flip_pair" if is_natural else row.get("variant", "base")
        key = (row["group_id"], row["template"], row["order"], axis, variant)
        groups[key].append((idx, row))

    records = []
    for key, pair_items in groups.items():
        if len(pair_items) != 2:
            raise ValueError(f"Expected 2 items per pair for {key}, found {len(pair_items)}.")
        pair_items.sort(key=lambda item: item[1]["target"])
        target_ids = [item[1]["target"] for item in pair_items]
        sub_matrix = scores[[item[0] for item in pair_items]][:, target_ids]
        dec = decompose_pair_matrix(sub_matrix)
        if dec["both_images_correct"] != dec["signal_exceeds_prior"]:
            # Only possible on exact boundary tie d0==0 or d1==0
            pass
        records.append({
            "group_id": key[0],
            "template": int(key[1]),
            "order": int(key[2]),
            "axis": "horizontal" if key[3] == 0 else "vertical",
            "variant": key[4],
            **dec,
        })

    summaries = {}
    for t in sorted({r["template"] for r in records}):
        for order in sorted({r["order"] for r in records}):
            for axis in sorted({r["axis"] for r in records}):
                sub = [r for r in records if r["template"] == t and r["order"] == order and r["axis"] == axis]
                if not sub:
                    continue
                mean_abs_b = float(np.mean([r["abs_prior_bias"] for r in sub]))
                mean_abs_sig = float(np.mean([r["abs_spatial_signal"] for r in sub]))
                median_psr = float(np.median([r["prior_to_signal_ratio"] for r in sub]))
                ratio_of_means = mean_abs_b / max(mean_abs_sig, 1e-12)
                summaries[f"t{t}_o{order}_{axis}"] = {
                    "n_pairs": len(sub),
                    "mean_abs_prior_bias_B": mean_abs_b,
                    "mean_abs_spatial_signal_delta": mean_abs_sig,
                    "mean_signed_prior_bias_B": float(np.mean([r["prior_bias_B"] for r in sub])),
                    "mean_signed_spatial_signal_delta": float(np.mean([r["spatial_signal_delta"] for r in sub])),
                    "aggregate_prior_to_signal_ratio": ratio_of_means,
                    "median_pair_psr": median_psr,
                    "prior_dominates_signal_percent": 100.0 * float(np.mean([r["abs_prior_bias"] > r["abs_spatial_signal"] for r in sub])),
                    "same_prediction_collapse_percent": 100.0 * float(np.mean([r["same_prediction_collapse"] for r in sub])),
                    "assignment_accuracy_percent": 100.0 * float(np.mean([r["assignment_correct"] for r in sub])),
                    "assignment_ci95_percent": cluster_ci(
                        [r["assignment_correct"] for r in sub],
                        [r["group_id"] for r in sub],
                        bootstrap_draws,
                        bootstrap_seed,
                    ),
                    "both_images_correct_percent": 100.0 * float(np.mean([r["both_images_correct"] for r in sub])),
                    "both_images_correct_ci95_percent": cluster_ci(
                        [r["both_images_correct"] for r in sub],
                        [r["group_id"] for r in sub],
                        bootstrap_draws,
                        bootstrap_seed,
                    ),
                }
    return {
        "stage": stage,
        "run_dir": run_path.relative_to(ROOT).as_posix() if run_path.is_relative_to(ROOT) else str(run_path),
        "conditions": summaries,
    }, records


def analyze_frozen_spatial_runs(output_path=None):
    """Run P2.1 Prior-to-Signal decomposition across all three frozen spatial benchmarks."""
    cfg = read_json(CONFIG_PATH)
    results = {}
    for name, rel_path in cfg["frozen_runs"].items():
        run_dir = ROOT / rel_path
        if run_dir.exists():
            summary, _ = analyze_run_pairs(
                run_dir,
                bootstrap_draws=cfg["bootstrap_draws"],
                bootstrap_seed=cfg["bootstrap_seed"],
            )
            results[name] = summary
    if output_path is not None:
        write_json(Path(output_path), results)
    return results

