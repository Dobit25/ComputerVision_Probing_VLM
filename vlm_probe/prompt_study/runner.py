"""Live GPU evaluator for the 7-template standard dataset suite and probing prompt suites (P3.1-P3.3)."""

import time
from pathlib import Path

import numpy as np

from ..dataset_adapters import load_dataset
from ..depth.common import ROOT, Engine, begin, finish, read_json, write_csv
from ..io_utils import sha256, write_json
from ..probing.stimuli import materialize
from .design import (
    CONFIG_PATH,
    build_counting_prompts,
    build_dataset_templates,
    build_negation_prompts,
    load_study_config,
)
from .metrics import (
    summarize_template_suite,
    text_embedding_ensemble,
)

PROTOCOL_PATH = ROOT / "docs/prompt_study_protocol.md"


def run_dataset_prompt_study(dataset_name, smoke=False, run_name=None):
    """Evaluate the 7 pre-registered templates + 3 ensemble methods on a standard dataset."""
    from torch.utils.data import Subset

    study_cfg = load_study_config()
    if dataset_name not in study_cfg["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    ds_entry = study_cfg["datasets"][dataset_name]
    ds_config_path = ROOT / ds_entry["config_path"]
    ds_cfg = read_json(ds_config_path)

    out, _, manifest = begin(
        f"prompt_{dataset_name}",
        smoke=smoke,
        inputs=[CONFIG_PATH, PROTOCOL_PATH, ds_config_path],
        run_name=run_name,
    )
    engine = Engine(out, {"model_config": ds_entry["config_path"]}, manifest)
    bundle = load_dataset(ROOT, ds_cfg, engine.adapter.preprocess)
    count = 200 if smoke else len(bundle.images)
    subset_ds = Subset(bundle.images, range(count))
    targets = np.array(bundle.targets[:count], dtype=int)

    templates_map = build_dataset_templates(dataset_name, bundle.classes, study_cfg)
    write_json(out / "templates.json", templates_map)

    started = time.perf_counter()
    identity = {
        "dataset": dataset_name,
        "count": count,
        "sample_ids_sha256": sha256_of_list(bundle.sample_ids[:count]),
    }
    image_features = engine.images(subset_ds, identity)

    text_embeddings = {}
    scores_by_template = {}
    for tid, prompts in templates_map.items():
        t_emb = engine.texts(prompts)
        text_embeddings[tid] = t_emb.numpy()
        scores_by_template[tid] = engine.scores(image_features, t_emb)

    # Text-embedding ensemble (CLIP-style two-stage L2 normalization)
    emb_ens_vec, alignment_norms = text_embedding_ensemble(list(text_embeddings.values()))
    emb_ens_scores = engine.scores(image_features, engine.torch.from_numpy(emb_ens_vec))

    suite_summary = summarize_template_suite(
        scores_dict=scores_by_template,
        targets=targets,
        baseline_key="T0_baseline",
        ensemble_scores_dict={"text_embedding_ensemble": emb_ens_scores},
        logit_scale=study_cfg["clip_logit_scale"],
        bootstrap_draws=study_cfg["bootstrap_draws"],
        bootstrap_seed=study_cfg["bootstrap_seed"],
    )
    suite_summary["dataset"] = dataset_name
    suite_summary["evaluated_samples"] = count
    suite_summary["num_classes"] = len(bundle.classes)
    suite_summary["mean_class_template_alignment_norm"] = float(alignment_norms.mean())
    suite_summary["min_class_template_alignment_norm"] = float(alignment_norms.min())
    suite_summary["evaluation_seconds"] = time.perf_counter() - started

    np.savez_compressed(
        out / "scores.npz",
        **scores_by_template,
        text_embedding_ensemble=emb_ens_scores,
        class_alignment_norms=alignment_norms,
    )
    write_json(out / "metrics.json", suite_summary)
    manifest.update(
        dataset=dataset_name,
        evaluated_samples=count,
        num_classes=len(bundle.classes),
        num_templates=len(templates_map),
    )
    finish(out, manifest)
    print(
        f"[{dataset_name}] Baseline: {suite_summary['baseline_accuracy_percent']:.2f}% | "
        f"Best single ({suite_summary['best_single_template']}): {suite_summary['best_single_accuracy_percent']:.2f}% | "
        f"Text-Emb Ensemble: {suite_summary['ensembles']['text_embedding_ensemble']['accuracy_percent']:.2f}%",
        flush=True,
    )
    return suite_summary


def run_probing_prompt_study(smoke=False, run_name=None):
    """Evaluate pre-registered multi-template suites on Counting (5 templates) and Negation (4 templates)."""
    from ..depth.common import ImageFiles

    study_cfg = load_study_config()
    out, _, manifest = begin(
        "prompt_probing",
        smoke=smoke,
        inputs=[CONFIG_PATH, PROTOCOL_PATH],
        run_name=run_name,
    )
    engine = Engine(out, {"model_config": study_cfg["model_config"]}, manifest)
    all_scenes = materialize(ROOT, seed=14)

    # 1. Counting (1,080 unique scenes)
    counting_rows = [r for r in all_scenes if r["task"] == "counting"]
    if smoke:
        counting_rows = counting_rows[:60]

    # 2. Negation (720 unique scenes, deduplicated from with_without condition)
    negation_rows = [r for r in all_scenes if r["task"] == "negation" and r["condition"] == "with_without"]
    if smoke:
        negation_rows = negation_rows[:60]

    results = {}
    for task_name, rows, builder in (
        ("counting", counting_rows, lambda r: build_counting_prompts(r["color"], r["shape"], study_cfg)),
        ("negation", negation_rows, lambda r: build_negation_prompts(r["target_color"], r["target_shape"], study_cfg)),
    ):
        im_dataset = ImageFiles(rows, engine.adapter.preprocess)
        im_feats = engine.images(
            im_dataset,
            {"task": f"prompt_study_{task_name}", "images": [r["image_sha256"] for r in rows]},
        )
        per_sample_templates = [builder(r) for r in rows]
        template_ids = list(per_sample_templates[0].keys())
        pool = list(dict.fromkeys(
            cap
            for tpl_map in per_sample_templates
            for caps in tpl_map.values()
            for cap in caps
        ))
        pool_idx = {c: i for i, c in enumerate(pool)}
        pool_embs = engine.texts(pool).numpy()
        all_sim = engine.scores(im_feats, engine.torch.from_numpy(pool_embs))

        n_samples = len(rows)
        n_cand = len(per_sample_templates[0][template_ids[0]])
        scores_dict = {
            tid: np.zeros((n_samples, n_cand), dtype=np.float32)
            for tid in template_ids
        }
        emb_ens_scores = np.zeros((n_samples, n_cand), dtype=np.float32)

        for i, tpl_map in enumerate(per_sample_templates):
            tpl_emb_list = []
            for tid in template_ids:
                idxs = [pool_idx[c] for c in tpl_map[tid]]
                scores_dict[tid][i] = all_sim[i, idxs]
                tpl_emb_list.append(pool_embs[idxs])
            ens_vec, _ = text_embedding_ensemble(tpl_emb_list)
            emb_ens_scores[i] = im_feats[i].numpy() @ ens_vec.T

        targets = np.array([int(r["target"]) for r in rows], dtype=int)
        groups = [r["group_id"] for r in rows]
        baseline_key = template_ids[0]
        results[task_name] = summarize_template_suite(
            scores_dict=scores_dict,
            targets=targets,
            baseline_key=baseline_key,
            ensemble_scores_dict={"text_embedding_ensemble": emb_ens_scores},
            groups=groups,
            logit_scale=study_cfg["clip_logit_scale"],
            bootstrap_draws=study_cfg["bootstrap_draws"],
            bootstrap_seed=study_cfg["bootstrap_seed"],
        )

    write_json(out / "metrics.json", results)
    finish(out, manifest)
    print(results, flush=True)
    return results


def sha256_of_list(items):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(list(items)).encode("utf-8")).hexdigest()

