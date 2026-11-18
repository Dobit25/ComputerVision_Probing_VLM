"""Controlled spatial interventions and common spatial/natural scoring."""
from collections import defaultdict

import numpy as np

from ..io_utils import sha256, write_json
from .common import ROOT, Engine, ImageFiles, begin, finish, cluster_ci, classification_stats, write_csv
from .design import spatial_images, spatial_trials


def compute_metrics(trials, scores, controls=None, blank_scores=None, config=None):
    cfg = config or {'bootstrap_draws': 2000, 'bootstrap_seed': 1402}
    y = np.array([r['target'] for r in trials])
    pred, correct, _, _, _, margin = classification_stats(scores, y)
    if len(set(r['sample_id'] for r in trials)) != len(trials):
        raise ValueError('Duplicate trial IDs.')
    summaries, records = {}, []
    for i, r in enumerate(trials):
        rec = {k: r[k] for k in ['sample_id', 'image_index', 'group_id', 'target', 'template', 'order', 'variant']}
        rec.update(prediction=int(pred[i]), correct=bool(correct[i]), signed_margin=float(margin[i]))
        if controls is not None:
            rec['control_correct'] = bool(np.argmax(controls[i]) == r['control_target'])
        if blank_scores is not None:
            rec['blank_prediction'] = int(np.argmax(blank_scores[i]))
        records.append(rec)
    variants = list(dict.fromkeys(r['variant'] for r in trials))
    for t in range(3):
        for order in range(2):
            for variant in ['all', *variants]:
                subset = [r for r in records if r['template'] == t and r['order'] == order and
                          (variant == 'all' or r['variant'] == variant)]
                matrix = np.zeros((4, 4), dtype=int)
                for r in subset:
                    matrix[r['target'], r['prediction']] += 1
                stats = {'n': len(subset), 'accuracy_percent': 100 * float(np.mean([r['correct'] for r in subset])),
                         'accuracy_ci95_percent': cluster_ci([r['correct'] for r in subset], [r['group_id'] for r in subset],
                                                            cfg['bootstrap_draws'], cfg['bootstrap_seed']),
                         'confusion': matrix.tolist(), 'prediction_counts': matrix.sum(0).tolist(),
                         'mean_signed_margin': float(np.mean([r['signed_margin'] for r in subset]))}
                if controls is not None:
                    ok = [r for r in subset if r['control_correct']]
                    stats.update(control_accuracy_percent=100 * len(ok)/len(subset),
                                 accuracy_given_control_correct_percent=100 * np.mean([r['correct'] for r in ok]) if ok else None)
                if blank_scores is not None:
                    stats['blank_prediction_agreement_percent'] = 100 * float(np.mean([r['prediction'] == r['blank_prediction'] for r in subset]))
                groups = defaultdict(list)
                for r in subset:
                    groups[(r['group_id'], r['variant'])].append(r)
                if all(len(g) == 4 and {r['target'] for r in g} == set(range(4)) for g in groups.values()):
                    stats['all_four_relations_correct_percent'] = 100 * float(np.mean([all(r['correct'] for r in g) for g in groups.values()]))
                    for name, labels in [('horizontal_pair', [0, 1]), ('vertical_pair', [2, 3])]:
                        values = [all(r['correct'] for r in g if r['target'] in labels) for g in groups.values()]
                        stats[name + '_both_correct_percent'] = 100 * float(np.mean(values))
                        stats[name + '_ci95_percent'] = cluster_ci(values, [k[0] for k in groups], cfg['bootstrap_draws'], cfg['bootstrap_seed'])
                summaries[f't{t}_o{order}_{variant}'] = stats
    equivalence, appearance, flips = {}, {}, {}
    for t in range(3):
        grouped = defaultdict(list)
        for r in records:
            if r['template'] == t:
                grouped[r['image_index']].append(r)
        if not all(len(g) == 2 for g in grouped.values()):
            raise ValueError('Missing equivalent-order trials.')
        same = [g[0]['prediction'] == g[1]['prediction'] for g in grouped.values()]
        both = [all(r['correct'] for r in g) for g in grouped.values()]
        equivalence[f't{t}'] = {'prediction_agreement_percent': 100 * float(np.mean(same)),
                                'both_orders_correct_percent': 100 * float(np.mean(both)),
                                'both_orders_ci95_percent': cluster_ci(both, [g[0]['group_id'] for g in grouped.values()], cfg['bootstrap_draws'], cfg['bootstrap_seed'])}
        for order in range(2):
            grouped = defaultdict(list)
            for r in records:
                if r['template'] == t and r['order'] == order:
                    grouped[(r['group_id'], r['target'])].append(r)
            if all(len(g) == 3 for g in grouped.values()):
                appearance[f't{t}_o{order}'] = {
                    'same_prediction_all_appearances_percent': 100 * float(np.mean([len({r['prediction'] for r in g}) == 1 for g in grouped.values()])),
                    'all_appearances_correct_percent': 100 * float(np.mean([all(r['correct'] for r in g) for g in grouped.values()]))}
            if set(variants) == {'original', 'horizontal_flip'}:
                by_source = defaultdict(list)
                for r in records:
                    if r['template'] == t and r['order'] == order:
                        by_source[r['group_id']].append(r)
                from .design import HFLIP
                values, consistent, horizontal_values = [], [], []
                for g in by_source.values():
                    if len(g) != 2:
                        raise ValueError('Missing image/flip partner.')
                    orig = next(r for r in g if r['variant'] == 'original')
                    flip = next(r for r in g if r['variant'] == 'horizontal_flip')
                    values.append(orig['correct'] and flip['correct'])
                    consistent.append(flip['prediction'] == HFLIP[orig['prediction']])
                    if orig['target'] in (0, 1):
                        horizontal_values.append(orig['correct'] and flip['correct'])
                flips[f't{t}_o{order}'] = {'both_correct_percent': 100 * float(np.mean(values)),
                    'both_correct_ci95_percent': cluster_ci(values, list(by_source), cfg['bootstrap_draws'], cfg['bootstrap_seed']),
                    'equivariant_prediction_percent': 100 * float(np.mean(consistent)),
                    'horizontal_both_correct_percent': 100 * float(np.mean(horizontal_values))}
    return {'conditions': summaries, 'equivalent_orders': equivalence, 'appearance': appearance, 'flips': flips}, records


def evaluate(out, cfg, manifest, images, medium):
    from PIL import Image
    engine = Engine(out, cfg, manifest)
    trials = spatial_trials(images, cfg, medium)
    for row in trials:
        im = images[row['image_index']]
        if 'control_captions' in im:
            row['control_captions'], row['control_target'] = im['control_captions'], im['control_target']
    write_json(out / 'images.json', images)
    write_json(out / 'trials.json', trials)
    has_control = 'control_captions' in trials[0]
    pool = list(dict.fromkeys(c for r in trials for c in r['captions'] + r.get('control_captions', [])))
    write_json(out / 'text_pool.json', pool)
    text_index = {c: i for i, c in enumerate(pool)}
    features = engine.images(ImageFiles(images, engine.adapter.preprocess),
                             {'images': [{k: r.get(k) for k in ['image_id', 'image_sha256', 'transform']} for r in images]})
    text = engine.texts(pool)
    all_scores = engine.scores(features, text)
    scores = np.array([[all_scores[r['image_index'], text_index[c]] for c in r['captions']] for r in trials], dtype=np.float32)
    arrays = {'scores': scores}
    if has_control:
        arrays['controls'] = np.array([[all_scores[r['image_index'], text_index[c]] for c in r['control_captions']] for r in trials], dtype=np.float32)
        backgrounds = sorted({im['background'] for im in images})
        with engine.torch.inference_mode():
            blank = engine.norm(engine.adapter.encode_images(engine.torch.stack([
                engine.adapter.preprocess(Image.new('RGB', (224, 224), (v,)*3)) for v in backgrounds]).to('cuda'))).cpu()
        blank_all = engine.scores(blank, text)
        arrays['blank_scores'] = np.array([[blank_all[backgrounds.index(images[r['image_index']]['background']), text_index[c]] for c in r['captions']] for r in trials], dtype=np.float32)
    np.savez_compressed(out / 'scores.npz', **arrays)
    metrics, records = compute_metrics(trials, config=cfg, **arrays)
    write_csv(out / 'predictions.csv', records)
    write_json(out / 'metrics.json', metrics)
    manifest.update(num_images=len(images), num_trials=len(trials), unique_caption_count=len(pool))
    finish(out, manifest)
    print({k: round(v['accuracy_percent'], 4) for k, v in metrics['conditions'].items() if k.endswith('_all')}, flush=True)


def run(smoke=False, run_name=None):
    out, cfg, manifest = begin('spatial', smoke, run_name=run_name)
    images = spatial_images(cfg)
    if len(images) != cfg['spatial']['expected_images'] or len({r['image_sha256'] for r in images}) != len(images):
        raise ValueError('Unexpected scene count or duplicate images.')
    if smoke:
        images = images[:36]  # Three complete groups, not an incomplete intervention group.
    evaluate(out, cfg, manifest, images, 'drawing')
