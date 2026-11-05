"""Paired metrics over all candidates, with class-stratified paired bootstrap."""

from ..metrics import classification_metrics


def summarize(targets, baseline, category, classes, draws=1000, seed=0):
    import numpy as np

    names = [c['class_name'] for c in classes]
    bm, bp, bc = classification_metrics(targets, baseline, names)
    cm, cp, cc = classification_metrics(targets, category, names)
    y, b, c = map(np.asarray, (targets, baseline, category))
    old, new = b == y, c == y
    fixed, broken = ~old & new, old & ~new
    delta = new.astype(int) - old.astype(int)
    rng = np.random.default_rng(seed)
    draws_total = np.zeros(draws)
    for target in sorted(set(targets)):
        values = delta[y == target]
        draws_total += values[rng.integers(0, len(values), size=(draws, len(values)))].sum(axis=1)
    paired = {'both_correct': int((old & new).sum()), 'both_wrong': int((~old & ~new).sum()),
              'wrong_to_correct': int(fixed.sum()), 'correct_to_wrong': int(broken.sum()),
              'prediction_changed': int((b != c).sum()), 'delta_percentage_points': float(100 * delta.mean()),
              'delta_ci95_percentage_points': (100 * np.quantile(draws_total / len(y), [.025, .975])).tolist(),
              'bootstrap': {'method': 'paired_stratified_by_fine_class', 'draws': draws, 'seed': seed}}
    per_class = []
    for i, (br, cr) in enumerate(zip(bp, cp)):
        selected = y == i
        support = int(selected.sum())
        per_class.append({'class_id': i, 'class_name': names[i], 'coarse_name': classes[i]['coarse_name'],
            'support': support, 'baseline_correct': br['correct'], 'category_correct': cr['correct'],
            'baseline_accuracy_percent': 100 * br['accuracy'] if support else None,
            'category_accuracy_percent': 100 * cr['accuracy'] if support else None,
            'delta_percentage_points': 100 * float(delta[selected].mean()) if support else None,
            'wrong_to_correct': int(fixed[selected].sum()), 'correct_to_wrong': int(broken[selected].sum())})
    coarse_ids = np.array([row['coarse_id'] for row in classes])
    groups = []
    for coarse_id in sorted(set(coarse_ids.tolist())):
        selected = coarse_ids[y] == coarse_id
        support = int(selected.sum())
        row = {'coarse_id': coarse_id, 'coarse_name': next(r['coarse_name'] for r in classes if r['coarse_id'] == coarse_id),
               'support': support, 'baseline_correct': int(old[selected].sum()), 'category_correct': int(new[selected].sum()),
               'delta_percentage_points': 100 * float(delta[selected].mean()) if support else None}
        for label, pred, ok in [('baseline', b, old), ('category', c, new)]:
            same_group = coarse_ids[pred] == coarse_ids[y]
            row[label + '_errors_within_group'] = int((selected & ~ok & same_group).sum())
            row[label + '_errors_across_groups'] = int((selected & ~ok & ~same_group).sum())
        groups.append(row)
    return {'baseline': bm, 'category': cm, 'paired': paired}, per_class, groups, {'baseline': bc, 'category': cc}
