"""Ensemble constructors and template sensitivity metrics with paired bootstrap CIs (P3.2-P3.3)."""

import numpy as np

from ..depth.common import cluster_ci, paired_class_ci


def l2_normalize(vectors, eps=1e-12):
    """L2-normalize a NumPy array along its last axis and verify unit norm."""
    arr = np.asarray(vectors, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values in embedding array.")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if (norms <= eps).any():
        raise ValueError("Zero-norm vector encountered during L2 normalization.")
    normalized = arr / norms
    return normalized.astype(np.float32)


def text_embedding_ensemble(template_embeddings):
    """CLIP-style two-stage L2-normalized text-embedding ensemble (P3.2 Method 1).

    Parameters
    ----------
    template_embeddings : list/sequence of (C, D) arrays or (M, C, D) array
        Each template's class embeddings. Every vector is first L2-normalized,
        then averaged across the M templates, and L2-normalized again.

    Returns
    -------
    ensembled : np.ndarray of shape (C, D), dtype float32
        Unit-norm ensembled class embeddings.
    alignment_norms : np.ndarray of shape (C,), dtype float64
        Pre-normalization mean vector norm ||bar{t}_c||_2 in (0, 1], measuring
        inter-template directional agreement for each class c.
    """
    stack = np.stack([l2_normalize(emb) for emb in template_embeddings], axis=0).astype(np.float64)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (M, C, D), got shape {stack.shape}.")
    mean_vec = stack.mean(axis=0)
    alignment_norms = np.linalg.norm(mean_vec, axis=-1)
    if (alignment_norms <= 1e-12).any():
        raise ValueError("Template embeddings cancelled out to zero norm.")
    ensembled = (mean_vec / alignment_norms[:, None]).astype(np.float32)
    return ensembled, alignment_norms


def logit_average_ensemble(scores_by_template):
    """Average cosine similarity / logit matrices across M templates (P3.2 Method 2a).

    Parameters
    ----------
    scores_by_template : list/sequence of (N, C) score matrices or (M, N, C) array

    Returns
    -------
    mean_scores : np.ndarray of shape (N, C), dtype float32
    """
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in scores_by_template], axis=0)
    if stack.ndim != 3 or not np.isfinite(stack).all():
        raise ValueError(f"Invalid score stack shape {stack.shape}.")
    return stack.mean(axis=0).astype(np.float32)


def probability_average_ensemble(scores_by_template, logit_scale=100.0):
    """Average temperature-scaled softmax probability distributions across M templates (P3.2 Method 2b).

    Parameters
    ----------
    scores_by_template : list/sequence of (N, C) score matrices
    logit_scale : float, default 100.0 (OpenAI CLIP ViT-B/32 learned exp(logit_scale) ~ 100.0)

    Returns
    -------
    mean_probs : np.ndarray of shape (N, C), dtype float32
    """
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in scores_by_template], axis=0)
    if stack.ndim != 3 or not np.isfinite(stack).all():
        raise ValueError(f"Invalid score stack shape {stack.shape}.")
    scaled = stack * float(logit_scale)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scaled)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return probs.mean(axis=0).astype(np.float32)


def _paired_ci(diff_bool_or_float, labels_or_groups, use_clusters=False, draws=2000, seed=1402):
    arr = np.asarray(diff_bool_or_float, dtype=np.float64)
    if use_clusters:
        return cluster_ci(arr, labels_or_groups, draws=draws, seed=seed)
    return paired_class_ci(arr, labels_or_groups, draws=draws, seed=seed)


def summarize_template_suite(
    scores_dict,
    targets,
    baseline_key,
    ensemble_scores_dict=None,
    groups=None,
    logit_scale=100.0,
    bootstrap_draws=2000,
    bootstrap_seed=1402,
):
    """Compute all P3.3 prompt sensitivity and ensembling statistics.

    Parameters
    ----------
    scores_dict : dict of {template_name: (N, C) score matrix}
    targets : array-like of shape (N,) with integer ground-truth indices
    baseline_key : str, key in `scores_dict` representing the baseline template
    ensemble_scores_dict : optional dict of additional ensemble score matrices
        (e.g., `{"text_embedding_ensemble": emb_scores}`). `logit_ensemble` and
        `prob_ensemble` over all keys in `scores_dict` are always computed automatically.
    groups : optional array-like of shape (N,) for cluster bootstrap (probing tasks).
        If None, stratified class paired bootstrap on `targets` is used.
    """
    y = np.asarray(targets, dtype=int)
    n = len(y)
    if baseline_key not in scores_dict:
        raise ValueError(f"baseline_key {baseline_key!r} not in scores_dict {list(scores_dict)}.")

    use_clusters = groups is not None
    strata = groups if use_clusters else y

    single_results = {}
    correct_vectors = {}
    for name, mat in scores_dict.items():
        m = np.asarray(mat, dtype=np.float64)
        if m.shape[0] != n:
            raise ValueError(f"Row count mismatch for {name}: {m.shape[0]} vs {n}.")
        preds = m.argmax(axis=1)
        corr = (preds == y).astype(np.float64)
        correct_vectors[name] = corr
        acc = 100.0 * float(corr.mean())
        single_results[name] = {
            "n": n,
            "correct": int(corr.sum()),
            "accuracy_percent": acc,
            "ci95_percent": _paired_ci(corr, strata, use_clusters=use_clusters, draws=bootstrap_draws, seed=bootstrap_seed),
        }

    for name in single_results:
        if name == baseline_key:
            single_results[name]["delta_vs_baseline_pp"] = 0.0
            single_results[name]["delta_vs_baseline_ci95_pp"] = [0.0, 0.0]
        else:
            diff = correct_vectors[name] - correct_vectors[baseline_key]
            single_results[name]["delta_vs_baseline_pp"] = 100.0 * float(diff.mean())
            single_results[name]["delta_vs_baseline_ci95_pp"] = _paired_ci(
                diff, strata, use_clusters=use_clusters, draws=bootstrap_draws, seed=bootstrap_seed
            )

    single_accs = [v["accuracy_percent"] for v in single_results.values()]
    best_name = max(single_results, key=lambda k: single_results[k]["accuracy_percent"])
    worst_name = min(single_results, key=lambda k: single_results[k]["accuracy_percent"])
    best_acc = single_results[best_name]["accuracy_percent"]
    worst_acc = single_results[worst_name]["accuracy_percent"]
    mean_acc = float(np.mean(single_accs))
    sd_acc = float(np.std(single_accs, ddof=1)) if len(single_accs) > 1 else 0.0
    spread_pp = float(best_acc - worst_acc)

    all_ensembles = {
        "logit_average_ensemble": logit_average_ensemble(list(scores_dict.values())),
        "probability_average_ensemble": probability_average_ensemble(list(scores_dict.values()), logit_scale=logit_scale),
    }
    if ensemble_scores_dict:
        all_ensembles.update(ensemble_scores_dict)

    ensemble_results = {}
    for ens_name, ens_mat in all_ensembles.items():
        em = np.asarray(ens_mat, dtype=np.float64)
        preds = em.argmax(axis=1)
        corr = (preds == y).astype(np.float64)
        acc = 100.0 * float(corr.mean())
        diff_base = corr - correct_vectors[baseline_key]
        diff_best = corr - correct_vectors[best_name]
        ensemble_results[ens_name] = {
            "n": n,
            "correct": int(corr.sum()),
            "accuracy_percent": acc,
            "ci95_percent": _paired_ci(corr, strata, use_clusters=use_clusters, draws=bootstrap_draws, seed=bootstrap_seed),
            "delta_vs_baseline_pp": 100.0 * float(diff_base.mean()),
            "delta_vs_baseline_ci95_pp": _paired_ci(
                diff_base, strata, use_clusters=use_clusters, draws=bootstrap_draws, seed=bootstrap_seed
            ),
            "delta_vs_best_single_pp": 100.0 * float(diff_best.mean()),
            "delta_vs_best_single_ci95_pp": _paired_ci(
                diff_best, strata, use_clusters=use_clusters, draws=bootstrap_draws, seed=bootstrap_seed
            ),
        }

    return {
        "num_templates": len(single_results),
        "baseline_template": baseline_key,
        "baseline_accuracy_percent": single_results[baseline_key]["accuracy_percent"],
        "best_single_template": best_name,
        "best_single_accuracy_percent": best_acc,
        "worst_single_template": worst_name,
        "worst_single_accuracy_percent": worst_acc,
        "mean_single_accuracy_percent": mean_acc,
        "sd_single_accuracy_percent": sd_acc,
        "spread_max_minus_min_pp": spread_pp,
        "single_templates": single_results,
        "ensembles": ensemble_results,
    }

