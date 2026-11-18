"""Offline multi-prompt sensitivity & ensembling diagnostics on frozen runs (P3.3-P3.4).

Computes exact prompt sensitivity (Best, Worst, Mean +/- SD, Spread) and
Logit / Probability Average Ensembles directly from the frozen score artifacts:
  1. Negation (`results/full_probing_clip_b32_v1/predictions.jsonl`):
     `with_without` vs `contains_does_not` vs Ensembles across 720 scenes,
     including present/absent polarity breakdown and group consistency!
  2. Synthetic Spatial (`results/full_depth_spatial_v1/scores.npz`):
     `T0`, `T1`, `T2` (forward `o0`, reversed `o1`, and 6-template ensemble)
     across 1,440 synthetic images (4-way and 2x2 pair assignment).
  3. Natural Spatial Exploratory (`results/full_depth_natural_v1/scores.npz`) &
     Natural Confirmation (`results/full_natural_confirmation_v1/scores.npz`):
     `T0`, `T1`, `T2` and Ensembles on 4-way and 2x2 horizontal assignment.
  4. CIFAR-100 Category Prompts (`results/full_depth_category_v1/scores.npz`):
     Sensitivity and ensembling across `baseline`, `category`, `neutral`, and
     the 5 deranged category prompts (`shuffle_101..105`).
"""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from ..depth.common import ROOT, read_json, cluster_ci
from ..depth.pairs import assignment_gap
from ..io_utils import write_json
from .metrics import (
    logit_average_ensemble,
    probability_average_ensemble,
    summarize_template_suite,
)


def analyze_negation_ensemble():
    """Evaluate prompt sensitivity and ensembling on the 720 unique negation scenes (P3.4)."""
    run_dir = ROOT / "results/full_probing_clip_b32_v1"
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if '"task": "negation"' in line
    ]
    by_scene = defaultdict(dict)
    meta_by_scene = {}
    for r in rows:
        # sample_id format: negation/{pair_index}/{layout}/{present}/{template}
        scene_key = "/".join(r["sample_id"].split("/")[:-1])
        template = r["condition"]
        by_scene[scene_key][template] = r["scores"]
        meta_by_scene[scene_key] = {
            "group_id": r["group_id"],
            "stratum": r["stratum"],
            "target": int(r["target"]),
        }

    scene_keys = sorted(by_scene)
    if len(scene_keys) != 720:
        raise ValueError(f"Expected 720 unique negation scenes, found {len(scene_keys)}.")

    scores_dict = {
        "with_without": np.array([by_scene[k]["with_without"] for k in scene_keys], dtype=np.float32),
        "contains_does_not": np.array([by_scene[k]["contains_does_not"] for k in scene_keys], dtype=np.float32),
    }
    targets = np.array([meta_by_scene[k]["target"] for k in scene_keys], dtype=int)
    groups = [meta_by_scene[k]["group_id"] for k in scene_keys]
    strata = [meta_by_scene[k]["stratum"] for k in scene_keys]

    suite = summarize_template_suite(
        scores_dict=scores_dict,
        targets=targets,
        baseline_key="with_without",
        groups=groups,
        bootstrap_draws=2000,
        bootstrap_seed=1402,
    )

    # Compute present/absent breakdown and group consistency (all scenes in group correct)
    all_matrices = {
        **scores_dict,
        "logit_average_ensemble": logit_average_ensemble(list(scores_dict.values())),
        "probability_average_ensemble": probability_average_ensemble(list(scores_dict.values())),
    }
    mechanistic = {}
    for cond_name, mat in all_matrices.items():
        preds = mat.argmax(axis=1)
        corr = preds == targets
        pres_mask = np.array([s == "present" for s in strata])
        abs_mask = ~pres_mask
        by_grp = defaultdict(list)
        for g, c in zip(groups, corr):
            by_grp[g].append(bool(c))
        grp_consistent = [all(vals) for vals in by_grp.values()]
        mechanistic[cond_name] = {
            "overall_accuracy_percent": 100.0 * float(corr.mean()),
            "present_accuracy_percent": 100.0 * float(corr[pres_mask].mean()),
            "absent_accuracy_percent": 100.0 * float(corr[abs_mask].mean()),
            "present_minus_absent_gap_pp": 100.0 * float(corr[pres_mask].mean() - corr[abs_mask].mean()),
            "scene_pair_group_consistency_percent": 100.0 * float(np.mean(grp_consistent)),
            "scene_pair_group_consistency_ci95_percent": cluster_ci(
                grp_consistent, list(by_grp.keys()), draws=2000, seed=1402
            ),
        }
    suite["mechanistic_breakdown"] = mechanistic
    return suite


def analyze_spatial_run_ensemble(run_dir):
    """Evaluate T0, T1, T2 sensitivity and ensembling on a spatial or confirmation run (P3.4)."""
    run_path = Path(run_dir)
    trials = read_json(run_path / "trials.json")
    with np.load(run_path / "scores.npz", allow_pickle=False) as archive:
        scores = archive["scores"]

    by_image = defaultdict(dict)
    image_meta = {}
    for idx, r in enumerate(trials):
        im_idx = r["image_index"]
        cond_key = f"t{r['template']}_o{r['order']}"
        by_image[im_idx][cond_key] = scores[idx]
        image_meta[im_idx] = {
            "group_id": r["group_id"],
            "target": int(r["target"]),
            "variant": r.get("variant", "original"),
        }

    im_indices = sorted(by_image)
    targets = np.array([image_meta[i]["target"] for i in im_indices], dtype=int)
    groups = [image_meta[i]["group_id"] for i in im_indices]

    forward_scores = {
        f"T{t}_forward": np.array([by_image[i][f"t{t}_o0"] for i in im_indices], dtype=np.float32)
        for t in range(3)
    }
    all_six_scores = {
        f"T{t}_{'forward' if o == 0 else 'reversed'}": np.array(
            [by_image[i][f"t{t}_o{o}"] for i in im_indices], dtype=np.float32
        )
        for t in range(3)
        for o in range(2)
    }

    four_way_forward = summarize_template_suite(
        scores_dict=forward_scores,
        targets=targets,
        baseline_key="T0_forward",
        groups=groups,
        bootstrap_draws=2000,
        bootstrap_seed=1402,
    )
    four_way_all_six = summarize_template_suite(
        scores_dict=all_six_scores,
        targets=targets,
        baseline_key="T0_forward",
        groups=groups,
        bootstrap_draws=2000,
        bootstrap_seed=1402,
    )

    # Also evaluate 2x2 horizontal pair assignment under T0, T1, T2 and their logit/prob ensembles
    horiz_pairs = defaultdict(list)
    for i in im_indices:
        meta = image_meta[i]
        if meta["target"] in (0, 1):
            pair_key = (meta["group_id"], meta["variant"] if meta["variant"] not in ("original", "horizontal_flip") else "flip_pair")
            horiz_pairs[pair_key].append(i)

    pair_keys = sorted(horiz_pairs)
    for pk in pair_keys:
        horiz_pairs[pk].sort(key=lambda idx: image_meta[idx]["target"])

    assignment_summary = {}
    ensemble_inputs = {
        **forward_scores,
        "T0_T1_T2_forward_logit_ensemble": logit_average_ensemble(list(forward_scores.values())),
        "T0_T1_T2_forward_prob_ensemble": probability_average_ensemble(list(forward_scores.values())),
        "all_6_orders_logit_ensemble": logit_average_ensemble(list(all_six_scores.values())),
    }
    for cond_name, mat in ensemble_inputs.items():
        assign_ok, both_ok, gaps = [], [], []
        grp_ids = []
        for pk in pair_keys:
            i0, i1 = horiz_pairs[pk]
            sub = np.stack([mat[im_indices.index(i0), :2], mat[im_indices.index(i1), :2]], axis=0)
            gap = assignment_gap(sub)
            dec = sub.argmax(axis=1)
            assign_ok.append(bool(gap > 0))
            both_ok.append(bool(dec[0] == 0 and dec[1] == 1))
            gaps.append(gap)
            grp_ids.append(pk[0])
        assignment_summary[cond_name] = {
            "n_horizontal_pairs": len(pair_keys),
            "assignment_accuracy_percent": 100.0 * float(np.mean(assign_ok)),
            "assignment_ci95_percent": cluster_ci(assign_ok, grp_ids, draws=2000, seed=1402),
            "both_images_correct_percent": 100.0 * float(np.mean(both_ok)),
            "mean_assignment_gap": float(np.mean(gaps)),
        }

    return {
        "n_images": len(im_indices),
        "four_way_forward_templates": four_way_forward,
        "four_way_all_six_templates": four_way_all_six,
        "horizontal_2x2_assignment": assignment_summary,
    }


def analyze_category_prompt_suite():
    """Analyze prompt sensitivity and ensembling across the 8 CIFAR-100 category conditions."""
    run_dir = ROOT / "results/full_depth_category_v1"
    samples = read_json(run_dir / "samples.json")
    targets = np.array([r["target_id"] for r in samples], dtype=int)
    with np.load(run_dir / "scores.npz", allow_pickle=False) as archive:
        all_scores = {k: archive[k] for k in archive.files}

    semantic_valid = {
        k: all_scores[k]
        for k in ("baseline", "category", "neutral")
        if k in all_scores
    }
    all_eight = dict(all_scores)

    return {
        "semantic_valid_3_templates": summarize_template_suite(
            scores_dict=semantic_valid,
            targets=targets,
            baseline_key="baseline",
            bootstrap_draws=2000,
            bootstrap_seed=1402,
        ),
        "all_8_conditions_including_wrong_category_shuffles": summarize_template_suite(
            scores_dict=all_eight,
            targets=targets,
            baseline_key="baseline",
            bootstrap_draws=2000,
            bootstrap_seed=1402,
        ),
    }


def analyze_saved_probing_ensembles(output_path=None):
    """Run all offline P3 sensitivity and ensembling analyses on frozen repository artifacts."""
    results = {
        "negation_probing_v1": analyze_negation_ensemble(),
        "synthetic_spatial_depth_v1": analyze_spatial_run_ensemble(ROOT / "results/full_depth_spatial_v1"),
        "natural_spatial_exploratory_v1": analyze_spatial_run_ensemble(ROOT / "results/full_depth_natural_v1"),
        "natural_spatial_confirmation_v1": analyze_spatial_run_ensemble(ROOT / "results/full_natural_confirmation_v1"),
        "cifar100_category_depth_v1": analyze_category_prompt_suite(),
    }
    if output_path is not None:
        write_json(Path(output_path), results)
    return results

