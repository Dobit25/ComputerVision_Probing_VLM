"""Pre-registered prompt templates for standard classification (31-condition matrix + 7-template suite) and probing (P3.1)."""

from collections import OrderedDict

from ..category_study.taxonomy import STUDY as CATEGORY_STUDY, load_classes as load_cifar100_taxonomy
from ..depth.common import ROOT, read_json
from ..probing.stimuli import NUMBERS

CONFIG_PATH = ROOT / "configs/prompt_study_v1.json"


def load_study_config():
    return read_json(CONFIG_PATH)


def build_n4_and_d4_templates(dataset_name, class_names, study_cfg=None):
    """Build the exact P3.1 31-condition template families (`N1..N4` for all 5 datasets + `D1..D4` for CIFAR-100).

    For DTD, keeps the ` texture` suffix consistent across `N1..N4` so `N1_photo` matches
    the frozen DTD baseline (`a photo of a {label} texture.`).
    For CIFAR-100, `D1..D4` append `, a type of {category}.` using the official 20 coarse groups.
    """
    cfg = study_cfg or load_study_config()
    if dataset_name not in cfg["datasets"]:
        raise ValueError(f"Unknown dataset {dataset_name!r}; expected one of {list(cfg['datasets'])}.")
    ds_cfg = cfg["datasets"][dataset_name]
    suffix = ds_cfg.get("label_suffix", "")
    clean_names = [name.replace("_", " ") + suffix for name in class_names]

    result = OrderedDict()
    for tid, pattern in cfg["n_templates"].items():
        result[tid] = [pattern.format(**{"class": cname}) for cname in clean_names]

    if ds_cfg.get("has_category_d_templates"):
        tax_rows, _ = load_cifar100_taxonomy(ROOT, CATEGORY_STUDY)
        if [r["class_name"] for r in tax_rows] != list(class_names):
            raise ValueError("CIFAR-100 class order mismatch with taxonomy.")
        categories = [r["category_text"] for r in tax_rows]
        for tid, pattern in cfg["d_templates_cifar100"].items():
            result[tid] = [
                pattern.format(**{"class": cname, "category": cat})
                for cname, cat in zip(clean_names, categories)
            ]
    return result


def build_dataset_templates(dataset_name, class_names, study_cfg=None):
    """Return an OrderedDict of {template_id: [prompt_per_class]} for a standard dataset.

    Guaranteed that T0_baseline (and N1_photo) matches the exact prompt formatting of `vlm_probe/evaluate.py`.
    """
    cfg = study_cfg or load_study_config()
    if dataset_name not in cfg["datasets"]:
        raise ValueError(f"Unknown dataset {dataset_name!r}; expected one of {list(cfg['datasets'])}.")
    ds_cfg = cfg["datasets"][dataset_name]
    common = cfg["common_templates"]
    suffix = ds_cfg.get("label_suffix", "")

    ordered_specs = OrderedDict([
        ("T0_baseline", common["T0_baseline"]),
        ("T1_minimal", common["T1_minimal"]),
        ("T2_clear", common["T2_clear"]),
        ("T3_centered", common["T3_centered"]),
        ("T4_domain", ds_cfg["domain_template"]),
        ("T5_bright", common["T5_bright"]),
        ("T6_cropped", common["T6_cropped"]),
    ])

    clean_names = [name.replace("_", " ") + suffix for name in class_names]
    prompts_by_template = OrderedDict()
    for tid, pattern in ordered_specs.items():
        rendered_names = [name.replace("_", " ") for name in class_names] if dataset_name == "dtd" and tid == "T4_domain" else clean_names
        prompts_by_template[tid] = [pattern.format(**{"class": cname}) for cname in rendered_names]
    return prompts_by_template


def build_counting_prompts(color, shape, study_cfg=None):
    """Return an OrderedDict of {template_id: [6 candidate captions for count 1..6]}."""
    cfg = study_cfg or load_study_config()
    templates = cfg["probing_templates"]["counting"]
    result = OrderedDict()
    for tid, pattern in templates.items():
        captions = []
        for idx, word in enumerate(NUMBERS):
            shape_plural = shape if idx == 0 else f"{shape}s"
            captions.append(pattern.format(word=word, color=color, shape_plural=shape_plural))
        result[tid] = captions
    return result


def build_negation_prompts(color, shape, study_cfg=None):
    """Return an OrderedDict of {template_id: [affirmative_caption, negative_caption]}."""
    cfg = study_cfg or load_study_config()
    templates = cfg["probing_templates"]["negation"]
    target = f"{color} {shape}"
    result = OrderedDict()
    for tid, (pos_pat, neg_pat) in templates.items():
        result[tid] = [pos_pat.format(target=target), neg_pat.format(target=target)]
    return result
