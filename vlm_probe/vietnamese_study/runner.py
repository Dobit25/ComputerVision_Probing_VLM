"""Live GPU evaluator for Cross-Lingual Vietnamese Prompting on CIFAR-100 and Probing (P4.1)."""

import time
from pathlib import Path

import numpy as np

from ..dataset_adapters import load_dataset
from ..depth.common import ROOT, Engine, begin, finish, read_json, write_csv
from ..io_utils import write_json
from ..extension_artifacts import begin_extension, save_classification
from ..prompt_study.metrics import summarize_template_suite, text_embedding_ensemble
from ..prompt_study.runner import sha256_of_list
from .bpe_diagnostics import analyze_bpe_tokenization
from .lexicon import build_cifar100_multilingual_prompts

CONFIG_PATH = ROOT / "configs/vietnamese_study_v1.json"
PROTOCOL_PATH = ROOT / "docs/vietnamese_study_protocol.md"


def run_cifar100_vietnamese_study(smoke=False, run_name=None):
    """Evaluate CIFAR-100 across en_baseline, vi_diacritics, vi_no_diacritics, and vi_en_bilingual."""
    from torch.utils.data import Subset

    cfg = read_json(CONFIG_PATH)
    model_cfg_path = ROOT / cfg["model_config"]
    model_cfg = read_json(model_cfg_path)

    out, _, manifest = begin_extension(
        "vietnamese_cifar100",
        CONFIG_PATH,
        smoke=smoke,
        inputs=[CONFIG_PATH, PROTOCOL_PATH, model_cfg_path],
        run_name=run_name,
    )
    engine = Engine(out, {"model_config": cfg["model_config"]}, manifest)
    bundle = load_dataset(ROOT, model_cfg, engine.adapter.preprocess)
    count = 200 if smoke else len(bundle.images)
    subset_ds = Subset(bundle.images, range(count))
    targets = np.array(bundle.targets[:count], dtype=int)

    conditions, class_records = build_cifar100_multilingual_prompts(bundle.classes)
    bpe_report = analyze_bpe_tokenization()
    write_json(out / "bpe_diagnostics.json", bpe_report)
    write_json(out / "multilingual_classes.json", class_records)

    started = time.perf_counter()
    identity = {
        "dataset": "cifar100",
        "count": count,
        "sample_ids_sha256": sha256_of_list(bundle.sample_ids[:count]),
    }
    image_features = engine.images(subset_ds, identity)

    text_embeddings = {}
    scores_dict = {}
    for cond_name, prompts in conditions.items():
        t_emb = engine.texts(prompts)
        text_embeddings[cond_name] = t_emb.numpy()
        scores_dict[cond_name] = engine.scores(image_features, t_emb)

    # Cross-lingual text embedding cosine similarity with English baseline per class
    en_emb = text_embeddings["en_baseline"]
    cross_lingual_sim = {}
    for cond_name, emb in text_embeddings.items():
        cos_per_class = (en_emb * emb).sum(axis=1)
        cross_lingual_sim[cond_name] = {
            "mean_cosine_with_en_baseline": float(cos_per_class.mean()),
            "sd_cosine_with_en_baseline": float(cos_per_class.std(ddof=1)),
            "min_cosine_with_en_baseline": float(cos_per_class.min()),
            "max_cosine_with_en_baseline": float(cos_per_class.max()),
        }

    # Bilingual + English + Vietnamese embedding ensemble
    en_vi_emb_ens, _ = text_embedding_ensemble([
        text_embeddings["en_baseline"],
        text_embeddings["vi_diacritics"],
        text_embeddings["vi_en_bilingual"],
    ])
    en_vi_ens_scores = engine.scores(image_features, engine.torch.from_numpy(en_vi_emb_ens))

    suite_summary = summarize_template_suite(
        scores_dict=scores_dict,
        targets=targets,
        baseline_key="en_baseline",
        ensemble_scores_dict={"en_vi_bilingual_embedding_ensemble": en_vi_ens_scores},
        bootstrap_draws=cfg["bootstrap_draws"],
        bootstrap_seed=cfg["bootstrap_seed"],
    )
    suite_summary["cross_lingual_text_alignment"] = cross_lingual_sim
    suite_summary["bpe_summary"] = {
        "prompts": bpe_report["cifar100_prompts_summary"],
        "labels": bpe_report["cifar100_labels_summary"],
    }
    suite_summary["evaluated_samples"] = count
    suite_summary["evaluation_seconds"] = time.perf_counter() - started

    np.savez_compressed(
        out / "scores.npz",
        **scores_dict,
        en_vi_bilingual_embedding_ensemble=en_vi_ens_scores,
    )
    write_json(out / "metrics.json", suite_summary)
    np.savez_compressed(out / 'text_embeddings.npz', **text_embeddings)
    save_classification(out, {**scores_dict, 'en_vi_bilingual_embedding_ensemble': en_vi_ens_scores},
        [{'sample_id': sid, 'target_id': int(y)} for sid, y in zip(bundle.sample_ids[:count], targets)],
        bundle.classes, 'en_baseline')
    manifest.update(evaluated_samples=count, conditions=list(conditions.keys()))
    finish(out, manifest)
    print(
        {k: round(v["accuracy_percent"], 2) for k, v in suite_summary["single_templates"].items()},
        flush=True,
    )
    return suite_summary

