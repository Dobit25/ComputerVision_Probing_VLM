"""Region / Half-Canvas Zero-Shot Spatial Scoring Mitigation (P2.2 - P2.3).

Splits each image into four half-canvases (left, right, top, bottom) and uses
single-object prompts ("a {medium} of {object}.") with the frozen CLIP model
to localize each object and infer their spatial relation without training.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..depth.common import ROOT, Engine, begin, finish, read_json, write_csv, cluster_ci
from ..depth.design import spatial_images
from ..depth.pairs import assignment_gap
from ..io_utils import sha256, write_json

CONFIG_PATH = ROOT / "configs/spatial_diagnostics_v1.json"
PROTOCOL_PATH = ROOT / "docs/spatial_diagnostics_protocol.md"


def crop_half_canvases(pil_image):
    """Return (left_half, right_half, top_half, bottom_half) PIL RGB crops."""
    im = pil_image.convert("RGB")
    w, h = im.size
    mid_x, mid_y = w // 2, h // 2
    return {
        "left": im.crop((0, 0, mid_x, h)),
        "right": im.crop((mid_x, 0, w, h)),
        "top": im.crop((0, 0, w, mid_y)),
        "bottom": im.crop((0, mid_y, w, h)),
    }


def relational_scores_from_regions(region_scores_a, region_scores_b):
    """Compute 4-way relational scores [left_of, right_of, above, below] from half-canvas scores.

    Parameters
    ----------
    region_scores_a : dict or array-like of shape (..., 4)
        Cosine similarity of object A prompt on (left, right, top, bottom) halves.
    region_scores_b : dict or array-like of shape (..., 4)
        Cosine similarity of object B prompt on (left, right, top, bottom) halves.

    Returns
    -------
    scores : np.ndarray of shape (..., 4)
        [S_left_of(A,B), S_right_of(A,B), S_above(A,B), S_below(A,B)]
        Guaranteed antisymmetric under swapping left<->right or top<->bottom,
        eliminating additive single-class prior bias.
    """
    if isinstance(region_scores_a, dict):
        sa = np.stack([region_scores_a[k] for k in ("left", "right", "top", "bottom")], axis=-1)
        sb = np.stack([region_scores_b[k] for k in ("left", "right", "top", "bottom")], axis=-1)
    else:
        sa = np.asarray(region_scores_a, dtype=np.float64)
        sb = np.asarray(region_scores_b, dtype=np.float64)
    if sa.shape != sb.shape or sa.shape[-1] != 4:
        raise ValueError("Expected matching (..., 4) arrays for (left, right, top, bottom).")

    # Horizontal contrast: how much more A is on the left vs right relative to B
    s_left = (sa[..., 0] - sa[..., 1]) - (sb[..., 0] - sb[..., 1])
    s_right = -s_left
    # Vertical contrast: how much more A is on the top vs bottom relative to B
    s_above = (sa[..., 2] - sa[..., 3]) - (sb[..., 2] - sb[..., 3])
    s_below = -s_above
    return np.stack([s_left, s_right, s_above, s_below], axis=-1)


def _box_half_overlap(bbox, width, height):
    """Compute fraction of a bounding box [x, y, w, h] falling in (left, right, top, bottom) halves."""
    x, y, bw, bh = [float(v) for v in bbox]
    x2, y2 = x + bw, y + bh
    area = max(bw * bh, 1e-9)
    mid_x, mid_y = width / 2.0, height / 2.0

    left_w = max(0.0, min(x2, mid_x) - max(x, 0.0))
    right_w = max(0.0, min(x2, float(width)) - max(x, mid_x))
    top_h = max(0.0, min(y2, mid_y) - max(y, 0.0))
    bot_h = max(0.0, min(y2, float(height)) - max(y, mid_y))
    return np.array([
        (left_w * bh) / area,
        (right_w * bh) / area,
        (bw * top_h) / area,
        (bw * bot_h) / area,
    ], dtype=np.float64)


def evaluate_geometric_upper_bound():
    """Evaluate the exact geometric upper bound of half-canvas cropping on synthetic and COCO natural sets.

    Checks how often an ideal single-object detector on half-canvases (left/right/top/bottom)
    correctly resolves the spatial relation on:
      1. Synthetic depth benchmark (1,440 images)
      2. Natural exploratory COCO benchmark (200 sources + 200 flips)
      3. Natural confirmation COCO benchmark (100 horizontal sources + 100 flips)
    and identifies why natural images still suffer when both objects lie in the same half-canvas!
    """
    depth_cfg = read_json(ROOT / "configs/probing_depth_v1.json")
    syn_images = spatial_images(depth_cfg, materialize=False)
    syn_correct = 0
    for im in syn_images:
        obj_a, obj_b = im["objects"]
        box_a = [obj_a["x"] - obj_a["radius"], obj_a["y"] - obj_a["radius"], 2 * obj_a["radius"], 2 * obj_a["radius"]]
        box_b = [obj_b["x"] - obj_b["radius"], obj_b["y"] - obj_b["radius"], 2 * obj_b["radius"], 2 * obj_b["radius"]]
        sa = _box_half_overlap(box_a, 224, 224)
        sb = _box_half_overlap(box_b, 224, 224)
        rel_scores = relational_scores_from_regions(sa, sb)
        if int(rel_scores.argmax()) == int(im["target"]):
            syn_correct += 1

    results = {
        "synthetic_depth_1440": {
            "n_images": len(syn_images),
            "four_way_accuracy_percent": 100.0 * syn_correct / len(syn_images),
            "straddles_midline_rate_percent": 100.0 * syn_correct / len(syn_images),
        }
    }

    for key, sel_path in (
        ("natural_exploratory_200", ROOT / "results/depth_natural_selection_v1/selected.json"),
        ("natural_confirmation_100", ROOT / "results/natural_confirmation_selection_v1/selected.json"),
    ):
        if not sel_path.exists():
            continue
        rows = read_json(sel_path)
        four_way_ok, two_way_ok, straddle_ok = 0, 0, 0
        same_half_cases = 0
        for r in rows:
            w, h = float(r["width"]), float(r["height"])
            sa = _box_half_overlap(r["bbox_a"], w, h)
            sb = _box_half_overlap(r["bbox_b"], w, h)
            rel_scores = relational_scores_from_regions(sa, sb)
            target = int(r["target"])
            if int(rel_scores.argmax()) == target:
                four_way_ok += 1
            if target in (0, 1) and int(rel_scores[:2].argmax()) == target:
                two_way_ok += 1
            # Check if centers of A and B lie on opposite sides of the image midline
            cx_a = r["bbox_a"][0] + r["bbox_a"][2] / 2.0
            cx_b = r["bbox_b"][0] + r["bbox_b"][2] / 2.0
            cy_a = r["bbox_a"][1] + r["bbox_a"][3] / 2.0
            cy_b = r["bbox_b"][1] + r["bbox_b"][3] / 2.0
            if target in (0, 1):
                straddles = (cx_a < w / 2.0 < cx_b) or (cx_b < w / 2.0 < cx_a)
            else:
                straddles = (cy_a < h / 2.0 < cy_b) or (cy_b < h / 2.0 < cy_a)
            if straddles:
                straddle_ok += 1
            else:
                same_half_cases += 1
        results[key] = {
            "n_sources": len(rows),
            "ideal_half_canvas_four_way_accuracy_percent": 100.0 * four_way_ok / len(rows),
            "ideal_half_canvas_horizontal_two_way_accuracy_percent": (
                100.0 * two_way_ok / max(1, sum(int(r["target"]) in (0, 1) for r in rows))
            ),
            "objects_straddle_image_midline_percent": 100.0 * straddle_ok / len(rows),
            "both_objects_in_same_half_count": same_half_cases,
            "both_objects_in_same_half_percent": 100.0 * same_half_cases / len(rows),
        }
    return results


def run_live_mitigation(smoke=False, run_name=None):
    """Run live CLIP ViT-B/32 half-canvas zero-shot mitigation on synthetic spatial stimuli."""
    from PIL import Image

    depth_cfg = read_json(ROOT / "configs/probing_depth_v1.json")
    diag_cfg = read_json(CONFIG_PATH)
    out, _, manifest = begin("spatial_mitigation", smoke=smoke, inputs=[CONFIG_PATH, PROTOCOL_PATH], run_name=run_name)
    images = spatial_images(depth_cfg, materialize=True)
    if smoke:
        images = images[:12 * diag_cfg["mitigation"]["smoke_groups"]]

    engine = Engine(out, depth_cfg, manifest)
    template = diag_cfg["mitigation"]["single_object_template"]
    prompts = list(dict.fromkeys(
        template.format(medium="drawing", object=im[side])
        for im in images
        for side in ("a", "b")
    ))
    prompt_idx = {p: i for i, p in enumerate(prompts)}
    text_features = engine.texts(prompts)

    region_names = ("left", "right", "top", "bottom")
    region_tensors = {k: [] for k in region_names}
    for im in images:
        path = ROOT / im["image_path"]
        if sha256(path) != im["image_sha256"]:
            raise ValueError(f"Checksum mismatch for {path}")
        with Image.open(path) as pil_im:
            crops = crop_half_canvases(pil_im)
            for k in region_names:
                region_tensors[k].append(engine.adapter.preprocess(crops[k]))

    region_scores_all = {}
    for k in region_names:
        batch_tensor = engine.torch.stack(region_tensors[k])
        feats = []
        with engine.torch.inference_mode():
            for i in range(0, len(batch_tensor), 32):
                feats.append(engine.norm(engine.adapter.encode_images(batch_tensor[i:i + 32].to("cuda"))).cpu())
        feats_cat = engine.torch.cat(feats)
        region_scores_all[k] = engine.scores(feats_cat, text_features)

    records = []
    rel_matrix = np.zeros((len(images), 4), dtype=np.float32)
    for idx, im in enumerate(images):
        pa = prompt_idx[template.format(medium="drawing", object=im["a"])]
        pb = prompt_idx[template.format(medium="drawing", object=im["b"])]
        sa = {k: region_scores_all[k][idx, pa] for k in region_names}
        sb = {k: region_scores_all[k][idx, pb] for k in region_names}
        r_scores = relational_scores_from_regions(sa, sb)
        rel_matrix[idx] = r_scores.astype(np.float32)
        pred4 = int(r_scores.argmax())
        target = int(im["target"])
        records.append({
            "image_id": im["image_id"],
            "group_id": im["group_id"],
            "variant": im["variant"],
            "target": target,
            "prediction": pred4,
            "correct": bool(pred4 == target),
            "two_way_axis_correct": bool(
                (r_scores[:2].argmax() == target) if target in (0, 1) else (r_scores[2:].argmax() + 2 == target)
            ),
            "score_left_of": float(r_scores[0]),
            "score_right_of": float(r_scores[1]),
            "score_above": float(r_scores[2]),
            "score_below": float(r_scores[3]),
        })

    by_group = defaultdict(list)
    for r in records:
        by_group[(r["group_id"], r["variant"])].append(r)

    horiz_both = [
        all(r["correct"] for r in g if r["target"] in (0, 1))
        for g in by_group.values()
    ]
    vert_both = [
        all(r["correct"] for r in g if r["target"] in (2, 3))
        for g in by_group.values()
    ]
    all_four = [all(r["correct"] for r in g) for g in by_group.values()]

    summary = {
        "n_images": len(records),
        "four_way_accuracy_percent": 100.0 * float(np.mean([r["correct"] for r in records])),
        "four_way_ci95_percent": cluster_ci(
            [r["correct"] for r in records],
            [r["group_id"] for r in records],
            diag_cfg["bootstrap_draws"],
            diag_cfg["bootstrap_seed"],
        ),
        "within_axis_two_way_accuracy_percent": 100.0 * float(np.mean([r["two_way_axis_correct"] for r in records])),
        "horizontal_pair_both_correct_percent": 100.0 * float(np.mean(horiz_both)),
        "vertical_pair_both_correct_percent": 100.0 * float(np.mean(vert_both)),
        "all_four_relations_correct_percent": 100.0 * float(np.mean(all_four)),
    }

    np.savez_compressed(out / "relational_scores.npz", scores=rel_matrix)
    write_csv(out / "predictions.csv", records)
    write_json(out / "metrics.json", summary)
    manifest.update(num_images=len(images), unique_single_object_prompts=len(prompts))
    finish(out, manifest)
    print(summary, flush=True)
    return summary

