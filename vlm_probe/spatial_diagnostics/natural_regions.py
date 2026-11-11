"""Region scoring on opened COCO sets; paired with frozen holistic T2 scores."""
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

from ..depth.common import ROOT, Engine, read_json, write_csv, cluster_ci, finish
from ..depth.natural import make_images
from ..depth.pairs import assignment_gap
from ..extension_artifacts import begin_extension, close_enough
from ..io_utils import sha256, write_json
from .quadrant_mitigation import crop_half_canvases, relational_scores_from_regions

CONFIG = ROOT / 'configs/natural_regions_v2.json'
REGIONS = ('left', 'right', 'top', 'bottom')


def baseline_canvas(image, size=224):
    """PIL bicubic shorter-edge resize, then rounded center crop, as baseline."""
    image = image.convert('RGB'); w, h = image.size
    shape = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)
    image = image.resize(shape, Image.Resampling.BICUBIC)
    x, y = round((shape[0] - size) / 2), round((shape[1] - size) / 2)
    return image.crop((x, y, x + size, y + size))


def holistic_scores(rows, entry):
    old = ROOT / entry['baseline']; images = read_json(old / 'images.json')
    by_id = {r['image_id']: i for i, r in enumerate(images)}
    trials = read_json(old / 'trials.json')
    with np.load(old / 'scores.npz', allow_pickle=False) as z: scores = z['scores']
    lookup = {r['image_index']: scores[i] for i, r in enumerate(trials)
              if r['template'] == 2 and r['order'] == 0}
    return np.stack([lookup[by_id[r['image_id']]] for r in rows])


def summarize(rows, scores, holistic, choices, cfg):
    n = len(rows)
    if scores.shape != (n, 4) or holistic.shape != (n, 4) or not np.isfinite(scores).all():
        raise ValueError('Invalid relational score shape')
    y = np.array([r['target'] for r in rows]); groups = [r['group_id'] for r in rows]
    if choices == 2 and not np.isin(y, [0, 1]).all(): raise ValueError('Nonhorizontal primary label')
    pred = scores[:, :choices].argmax(1); base = holistic[:, :choices].argmax(1)
    correct = pred == y; base_ok = base == y
    result = {'n_images': n, 'n_sources': len(set(groups)), 'primary_choices': choices,
              'primary_accuracy_percent': float(100 * correct.mean()),
              'primary_ci95_percent': cluster_ci(correct, groups, cfg['bootstrap_draws'], cfg['bootstrap_seed']),
              'holistic_accuracy_percent': float(100 * base_ok.mean()),
              'paired_delta_pp': float(100 * (correct.astype(float) - base_ok).mean()),
              'paired_delta_ci95_pp': cluster_ci(correct.astype(float) - base_ok, groups, cfg['bootstrap_draws'], cfg['bootstrap_seed']),
              'four_way_accuracy_percent': float(100 * (scores.argmax(1) == y).mean()),
              'exact_top_ties': int((np.sort(scores[:, :choices], axis=1)[:, -1] == np.sort(scores[:, :choices], axis=1)[:, -2]).sum())}
    grouped = defaultdict(list)
    for i, r in enumerate(rows): grouped[r['group_id']].append(i)
    if any(len(g) != 2 for g in grouped.values()): raise ValueError('Expected original/mirror pairs')
    result['both_images_correct_percent'] = float(100 * np.mean([correct[g].all() for g in grouped.values()]))
    pair_diff = [float(correct[g].all()) - float(base_ok[g].all()) for g in grouped.values()]
    result['holistic_both_images_correct_percent'] = float(100 * np.mean([base_ok[g].all() for g in grouped.values()]))
    result['both_correct_paired_delta_pp'] = float(100 * np.mean(pair_diff))
    result['both_correct_paired_delta_ci95_pp'] = cluster_ci(pair_diff, list(grouped), cfg['bootstrap_draws'], cfg['bootstrap_seed'])
    matrix = np.zeros((choices, choices), dtype=int)
    for target, prediction in zip(y, pred): matrix[target, prediction] += 1
    result['primary_confusion'] = matrix.tolist()
    result['by_relation'] = {str(c): {'n': int((y == c).sum()),
        'accuracy_percent': float(100 * correct[y == c].mean()) if (y == c).any() else None}
        for c in range(choices)}
    pair_records = []
    for group, ix in grouped.items():
        if {int(y[i]) for i in ix} != {0, 1}: continue
        ix.sort(key=lambda i: y[i]); gap = assignment_gap(scores[ix, :2].astype(np.float64))
        pair_records.append({'group_id': group, 'gap': gap, 'correct': gap > 0,
                             'exact_tie': gap == 0, 'both_images_correct': bool(correct[ix].all())})
    if pair_records:
        result['horizontal_assignment'] = {'n_sources': len(pair_records),
            'accuracy_percent': float(100 * np.mean([r['correct'] for r in pair_records])),
            'ci95_percent': cluster_ci([r['correct'] for r in pair_records], [r['group_id'] for r in pair_records], cfg['bootstrap_draws'], cfg['bootstrap_seed']),
            'exact_ties': sum(r['exact_tie'] for r in pair_records)}
    predictions = [{'image_id': r['image_id'], 'group_id': r['group_id'], 'target': int(y[i]),
                    'prediction': int(pred[i]), 'holistic_prediction': int(base[i]),
                    'correct': bool(correct[i]), 'holistic_correct': bool(base_ok[i])}
                   for i, r in enumerate(rows)]
    return result, predictions, pair_records


def run(set_name, smoke=False, run_name=None):
    import torch
    from torch.utils.data import DataLoader, Dataset
    cfg = read_json(CONFIG); entry = cfg['sets'][set_name]
    selected = read_json(ROOT / entry['selection'])
    if smoke: selected = selected[:8]
    rows = make_images(selected)
    out, _, manifest = begin_extension('natural_region_' + set_name, CONFIG,
        inputs=[ROOT / entry['selection'], ROOT / entry['baseline'] / 'manifest.json'],
        smoke=smoke, run_name=run_name)
    engine = Engine(out, cfg, manifest)
    prompts = list(dict.fromkeys(cfg['object_template'].format(object=r[side])
                                for r in rows for side in ('a', 'b')))
    pool = {p: i for i, p in enumerate(prompts)}
    text = engine.texts(prompts)

    class Crops(Dataset):
        def __len__(self): return 4 * len(rows)
        def __getitem__(self, i):
            row = rows[i // 4]; path = ROOT / row['image_path']
            if sha256(path) != row['image_sha256']: raise ValueError('Changed source image')
            with Image.open(path) as im: image = baseline_canvas(im, cfg['canvas_size'])
            if row['transform'] == 'horizontal_flip': image = ImageOps.mirror(image)
            return engine.adapter.preprocess(crop_half_canvases(image)[REGIONS[i % 4]])

    feats = []
    for batch in DataLoader(Crops(), batch_size=32, shuffle=False, num_workers=0):
        with torch.inference_mode(): feats.append(engine.norm(engine.adapter.encode_images(batch.to('cuda'))).cpu())
    all_scores = engine.scores(torch.cat(feats), text).reshape(len(rows), 4, len(prompts))
    regional = np.empty((len(rows), 4, 2), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, side in enumerate(('a', 'b')):
            regional[i, :, j] = all_scores[i, :, pool[cfg['object_template'].format(object=r[side])]]
    relational = relational_scores_from_regions(regional[:, :, 0], regional[:, :, 1]).astype(np.float32)
    base = holistic_scores(rows, entry)
    metrics, predictions, pairs = summarize(rows, relational, base, entry['primary_choices'], cfg)
    write_json(out / 'samples.json', rows); write_json(out / 'prompts.json', prompts)
    write_json(out / 'metrics.json', metrics)
    write_csv(out / 'predictions.csv', predictions)
    if pairs: write_csv(out / 'pairs.csv', pairs)
    np.savez_compressed(out / 'scores.npz', regions=regional, relational=relational, holistic=base)
    manifest.update(set_name=set_name, num_sources=len(selected), num_images=len(rows))
    finish(out, manifest); print(metrics, flush=True)


def audit(directory):
    cfg = read_json(directory / 'study_config.json'); m = read_json(directory / 'manifest.json')
    entry = cfg['sets'][m['set_name']]
    rows = make_images(read_json(ROOT / entry['selection'])[:m['num_sources']])
    close_enough(read_json(directory / 'samples.json'), rows)
    with np.load(directory / 'scores.npz', allow_pickle=False) as z:
        regional, relational, base = z['regions'], z['relational'], z['holistic']
    if regional.shape != (len(rows), 4, 2): raise ValueError('Wrong region shape')
    close_enough(relational.tolist(), relational_scores_from_regions(regional[:, :, 0], regional[:, :, 1]).astype(np.float32).tolist())
    if not np.array_equal(base, holistic_scores(rows, entry)): raise ValueError('Holistic reference changed')
    metrics, predictions, pairs = summarize(rows, relational, base, entry['primary_choices'], cfg)
    close_enough(read_json(directory / 'metrics.json'), metrics)
    from ..depth.common import csv_rows
    if [{k: str(v) for k, v in r.items()} for r in predictions] != csv_rows(directory / 'predictions.csv'):
        raise ValueError('Prediction mismatch')
    if pairs and [{k: str(v) for k, v in r.items()} for r in pairs] != csv_rows(directory / 'pairs.csv'):
        raise ValueError('Pair mismatch')
    return {'metrics_recomputed': True, 'region_to_relational_scores_recomputed': True,
            'n': len(rows), 'metrics': metrics}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--set', choices=['exploratory', 'confirmation'], required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--run-name', required=True)
    a = p.parse_args(); run(a.set, a.smoke, a.run_name)
