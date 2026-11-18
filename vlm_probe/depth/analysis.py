"""Exploratory diagnostics using immutable historical score artifacts."""
from collections import Counter, defaultdict

import numpy as np

from ..io_utils import write_json
from .common import ROOT, begin, finish, read_json, csv_rows, write_csv, classification_stats


def score_diagnostics(scores, labels, classes):
    baseline = classification_stats(scores['baseline'], labels)
    category = classification_stats(scores['category'], labels)
    coarse = np.array([r['coarse_id'] for r in classes])
    labels = np.asarray(labels)
    rows = []
    for i, y in enumerate(labels):
        b, c = int(baseline[0][i]), int(category[0][i])
        fixed = int(baseline[3][i])
        rows.append({'test_index': i, 'class_id': int(y), 'class_name': classes[y]['class_name'],
                     'baseline_prediction': b, 'category_prediction': c,
                     'transition': ('both_correct' if b == y else 'wrong_to_correct') if c == y else
                                   ('correct_to_wrong' if b == y else 'both_wrong'),
                     'baseline_target_score': float(baseline[2][i]), 'category_target_score': float(category[2][i]),
                     'target_score_delta': float(category[2][i]-baseline[2][i]),
                     'baseline_rival_score': float(baseline[4][i]), 'category_rival_score': float(category[4][i]),
                     'rival_score_delta': float(category[4][i]-baseline[4][i]),
                     'baseline_margin': float(baseline[5][i]), 'category_margin': float(category[5][i]),
                     'margin_delta': float(category[5][i]-baseline[5][i]),
                     'baseline_rival_id': fixed, 'category_rival_id': int(category[3][i]),
                     'fixed_rival_score_delta': float(scores['category'][i, fixed]-scores['baseline'][i, fixed]),
                     'baseline_error_type': 'correct' if b == y else ('within' if coarse[b] == coarse[y] else 'across'),
                     'category_error_type': 'correct' if c == y else ('within' if coarse[c] == coarse[y] else 'across')})
    summary, by_class = [], []
    keys = ['target_score_delta', 'rival_score_delta', 'margin_delta', 'fixed_rival_score_delta']
    for transition in ['all', 'both_correct', 'both_wrong', 'wrong_to_correct', 'correct_to_wrong']:
        selected = rows if transition == 'all' else [r for r in rows if r['transition'] == transition]
        summary.append({'transition': transition, 'n': len(selected),
                        **{f'mean_{key}': float(np.mean([r[key] for r in selected])) if selected else None for key in keys},
                        'target_score_increased_fraction': float(np.mean([r['target_score_delta'] > 0 for r in selected])) if selected else None})
    for c in classes:
        selected = [r for r in rows if r['class_id'] == c['class_id']]
        if not selected:
            continue
        by_class.append({'class_id': c['class_id'], 'class_name': c['class_name'], 'coarse_name': c['coarse_name'],
                         'support': len(selected), **{f'mean_{k}': float(np.mean([r[k] for r in selected])) for k in keys},
                         'baseline_accuracy_percent': 100 * float(np.mean([r['baseline_prediction'] == c['class_id'] for r in selected])),
                         'category_accuracy_percent': 100 * float(np.mean([r['category_prediction'] == c['class_id'] for r in selected]))})
    return rows, summary, by_class


def run(run_name=None):
    old = ROOT / 'results/full_probing_clip_b32_v1'
    cat = ROOT / 'results/full_cifar100_category_v1'
    inputs = [old / 'manifest.json', old / 'predictions.jsonl', old / 'samples.json',
              cat / 'manifest.json', cat / 'scores.npz', cat / 'samples.json', cat / 'classes.json']
    out, cfg, manifest = begin('historical', inputs=inputs, run_name=run_name)
    samples = {r['sample_id']: r for r in read_json(old / 'samples.json')}
    import json
    rows = [json.loads(line) for line in (old / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    aggregate = {}
    breakdown = []
    for task in ['spatial', 'counting', 'negation']:
        selected = [r for r in rows if r['task'] == task]
        k = len(selected[0]['scores'])
        matrix = np.zeros((k, k), dtype=int)
        groups = defaultdict(list)
        for r in selected:
            if r['prediction'] != int(np.argmax(r['scores'])):
                raise ValueError('Historical prediction differs from scores.')
            matrix[r['target'], r['prediction']] += 1
            groups[r['group_id']].append(r)
        aggregate[task] = {'n': len(selected), 'confusion': matrix.tolist(),
                           'prediction_counts': matrix.sum(0).tolist(),
                           'accuracy_percent': 100 * float(np.trace(matrix) / len(selected)),
                           'all_group_correct_percent': 100 * np.mean([all(r['correct'] for r in g) for g in groups.values()])}
        if task == 'counting':
            aggregate[task]['mean_signed_count_error'] = float(np.mean([r['prediction']-r['target'] for r in selected]))
        factors = {'spatial': ['condition', 'a_color', 'a_shape', 'b_color', 'b_shape'],
                   'counting': ['condition', 'number', 'color', 'shape'],
                   'negation': ['condition', 'stratum']}[task]
        for factor in factors:
            splits = defaultdict(list)
            for r in selected:
                sample = samples[r['sample_id']]
                if factor.startswith(('a_', 'b_')):
                    value = sample['objects'][0 if factor[0] == 'a' else 1][factor[2:]]
                else:
                    value = sample.get(factor, r.get(factor))
                splits[str(value)].append(r)
            for value, group in sorted(splits.items()):
                breakdown.append({'task': task, 'factor': factor, 'value': value, 'n': len(group),
                                  'accuracy_percent': 100 * np.mean([r['correct'] for r in group]),
                                  'mean_signed_margin': np.mean([r['signed_margin'] for r in group])})
    labels = [r['target_id'] for r in read_json(cat / 'samples.json')]
    with np.load(cat / 'scores.npz', allow_pickle=False) as archive:
        score_rows, shifts, classes = score_diagnostics(dict(archive), labels, read_json(cat / 'classes.json'))
    write_csv(out / 'category_sample_scores.csv', score_rows)
    write_csv(out / 'category_class_scores.csv', classes)
    write_csv(out / 'category_transition_scores.csv', shifts)
    write_csv(out / 'probe_breakdown.csv', breakdown)
    write_json(out / 'metrics.json', {'probing': aggregate, 'category_score_shifts': shifts})
    finish(out, manifest)
    print(json.dumps({'probing': aggregate, 'category_score_shifts': shifts}, indent=2), flush=True)
