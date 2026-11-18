"""Exploratory 2x2 assignment diagnostic; explicitly separate from primary 4-way tests."""
from collections import defaultdict
import numpy as np
from .common import read_json, cluster_ci


def assignment_gap(matrix):
    """Diagonal assignment minus swapped assignment; text/image additive biases cancel."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError('Expected a two-image, two-caption score matrix.')
    return float(matrix[0, 0] + matrix[1, 1] - matrix[0, 1] - matrix[1, 0])


def analyze(out):
    trials = read_json(out / 'trials.json')
    config = read_json(out / 'config.json')
    with np.load(out / 'scores.npz', allow_pickle=False) as z:
        scores = z['scores']
    natural = read_json(out / 'manifest.json')['stage'] == 'natural'
    groups = defaultdict(list)
    for i, row in enumerate(trials):
        if natural and row['target'] not in (0, 1):
            continue
        axis = row['target'] // 2
        key = (row['group_id'], row['template'], row['order'], axis,
               'flip_pair' if natural else row['variant'])
        groups[key].append((i, row))
    records = []
    for key, pairs in groups.items():
        if len(pairs) != 2:
            raise ValueError('Missing 2x2 pair.')
        pairs.sort(key=lambda p: p[1]['target'])
        target_ids = [p[1]['target'] for p in pairs]
        matrix = scores[[p[0] for p in pairs]][:, target_ids]
        gap = assignment_gap(matrix)
        decisions = matrix.argmax(1)
        records.append({'group_id': key[0], 'template': key[1], 'order': key[2],
                        'axis': 'horizontal' if key[3] == 0 else 'vertical', 'variant': key[4],
                        'assignment_gap': gap, 'assignment_correct': gap > 0,
                        'near_tie': abs(gap) <= 1e-6,
                        'two_way_correct_count': int((decisions == np.array([0, 1])).sum()),
                        'two_way_both_correct': bool(np.array_equal(decisions, [0, 1]))})
    summary = {}
    for t in range(3):
        for order in range(2):
            for axis in sorted({r['axis'] for r in records}):
                subset = [r for r in records if r['template'] == t and r['order'] == order and r['axis'] == axis]
                summary[f't{t}_o{order}_{axis}'] = {'pairs': len(subset),
                    'assignment_accuracy_percent': 100 * float(np.mean([r['assignment_correct'] for r in subset])),
                    'assignment_ci95_percent': cluster_ci([r['assignment_correct'] for r in subset], [r['group_id'] for r in subset], config['bootstrap_draws'], config['bootstrap_seed']),
                    'two_way_accuracy_percent': 50 * float(np.mean([r['two_way_correct_count'] for r in subset])),
                    'two_way_both_correct_percent': 100 * float(np.mean([r['two_way_both_correct'] for r in subset])),
                    'mean_assignment_gap': float(np.mean([r['assignment_gap'] for r in subset])),
                    'near_tie_count': sum(r['near_tie'] for r in subset)}
    return summary, records
