"""Recomputable probe metrics; bootstrap units respect paired scene groups."""

from collections import defaultdict
import math


def score_row(sample, scores, control_scores=None, global_prediction_id=None):
    if len(scores) != len(sample["captions"]) or len(scores) < 2:
        raise ValueError("Caption/score count mismatch.")
    if any(not math.isfinite(x) or not -1.00001 <= x <= 1.00001 for x in scores):
        raise ValueError("Invalid cosine scores.")
    target = sample["target"]
    if type(target) is not int or not 0 <= target < len(scores):
        raise ValueError("Invalid target.")
    prediction = max(range(len(scores)), key=scores.__getitem__)
    ordered = sorted(scores, reverse=True)
    result = {"sample_id": sample["sample_id"], "group_id": sample["group_id"],
              "task": sample["task"], "condition": sample["condition"], "stratum": sample["stratum"],
              "target": target, "scores": scores, "prediction": prediction,
              "correct": prediction == target,
              "signed_margin": scores[target] - max(x for i, x in enumerate(scores) if i != target),
              "near_tie": ordered[0] - ordered[1] <= 1e-6,
              "exact_tie": ordered[0] == ordered[1]}
    for key in ("color", "shape", "number", "global_target_id", "candidate_class_ids"):
        if key in sample:
            result[key] = sample[key]
    if sample["task"] == "counting":
        result["absolute_count_error"] = abs(prediction - target)
    if sample["task"] == "fine_grained":
        if global_prediction_id is None:
            raise ValueError("Fine-grained requires global prediction for parity audit.")
        result["global_prediction_id"] = global_prediction_id
    if "control_captions" in sample:
        if control_scores is None or len(control_scores) != len(sample["control_captions"]):
            raise ValueError("Missing control scores.")
        if any(not math.isfinite(x) or not -1.00001 <= x <= 1.00001 for x in control_scores):
            raise ValueError("Invalid control scores.")
        target_control = sample["control_target"]
        prediction_control = max(range(len(control_scores)), key=control_scores.__getitem__)
        result.update(control_scores=control_scores, control_prediction=prediction_control,
                      control_correct=prediction_control == target_control)
    return result


def aggregate(rows):
    if not rows:
        return {"num_samples": 0}
    correct = sum(r["correct"] for r in rows)
    result = {"num_samples": len(rows), "correct": correct, "accuracy": correct / len(rows),
              "accuracy_percent": 100 * correct / len(rows),
              "mean_signed_margin": sum(r["signed_margin"] for r in rows) / len(rows),
              "near_tie_count": sum(r["near_tie"] for r in rows),
              "exact_tie_count": sum(r["exact_tie"] for r in rows)}
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    result["num_groups"] = len(groups)
    result["all_rows_correct_group_rate"] = sum(all(r["correct"] for r in group) for group in groups.values()) / len(groups)
    if "control_correct" in rows[0]:
        control_ok = [r for r in rows if r["control_correct"]]
        result["control_accuracy"] = len(control_ok) / len(rows)
        result["control_correct_samples"] = len(control_ok)
        result["accuracy_given_control_correct"] = sum(r["correct"] for r in control_ok) / len(control_ok) if control_ok else None
    if rows[0]["task"] == "counting":
        result["mean_absolute_count_error"] = sum(r["absolute_count_error"] for r in rows) / len(rows)
    if rows[0]["task"] == "negation":
        result["affirmative_choice_rate"] = sum(r["prediction"] == 0 for r in rows) / len(rows)
    return result


def bootstrap_ci(rows, draws=1000, seed=0):
    import numpy as np

    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row["correct"])
    counts = np.array([len(group) for group in groups.values()])
    correct = np.array([sum(group) for group in groups.values()])
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, draws, 50):
        indices = rng.integers(0, len(groups), size=(min(50, draws - start), len(groups)))
        values.extend((correct[indices].sum(axis=1) / counts[indices].sum(axis=1)).tolist())
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def summarize(rows):
    summary = {}
    for task in ("counting", "spatial", "negation", "fine_grained"):
        selected = [r for r in rows if r["task"] == task]
        if not selected:
            continue
        stats = aggregate(selected)
        stats["accuracy_ci95_group_bootstrap"] = bootstrap_ci(selected)
        stats["chance_accuracy"] = {"counting": 1 / 6, "spatial": 0.25, "negation": 0.5, "fine_grained": 0.2}[task]
        for field in ("condition", "stratum", "color", "shape"):
            if field not in selected[0]:
                continue
            values = sorted({r[field] for r in selected})
            stats[f"by_{field}"] = {value: aggregate([r for r in selected if r[field] == value]) for value in values}
        if task == "fine_grained":
            matrix = [[0] * 100 for _ in range(100)]
            for row in selected:
                matrix[row["global_target_id"]][row["candidate_class_ids"][row["prediction"]]] += 1
            stats["confusion_global_fine_ids"] = matrix
        else:
            size = len(selected[0]["scores"])
            matrix = [[0] * size for _ in range(size)]
            for row in selected:
                matrix[row["target"]][row["prediction"]] += 1
            stats["confusion"] = matrix
        summary[task] = stats
    return summary
