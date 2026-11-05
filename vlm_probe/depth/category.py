"""Eight paired category conditions; no fitting or outcome-driven prompt choice."""
import numpy as np

from ..dataset_adapters import load_dataset
from ..io_utils import sha256, write_json
from .common import ROOT, begin, finish, read_json, Engine, classification_stats, paired_class_ci, write_csv
from .design import category_prompts
from .analysis import score_diagnostics


def compute_metrics(scores, labels, classes, cfg):
    y = np.asarray(labels, dtype=int)
    coarse = np.array([r['coarse_id'] for r in classes])
    stats = {name: classification_stats(matrix, y) for name, matrix in scores.items()}
    if any(matrix.shape != (len(y), 100) for matrix in scores.values()):
        raise ValueError('All category conditions must include exactly 100 candidates.')
    conditions, per_class = {}, []
    base = stats['baseline'][1].astype(float)
    for name, (pred, correct, true, rival, rival_score, margin) in stats.items():
        delta = correct.astype(float) - base
        conditions[name] = {
            'n': len(y), 'correct': int(correct.sum()), 'accuracy_percent': 100 * float(correct.mean()),
            'macro_accuracy_percent': 100 * float(np.mean([correct[y == c].mean() for c in np.unique(y)])),
            'delta_vs_baseline_pp': 100 * float(delta.mean()),
            'delta_vs_baseline_ci95_pp': paired_class_ci(delta, y, cfg['bootstrap_draws'], cfg['bootstrap_seed']),
            'wrong_to_correct': int(((base == 0) & correct).sum()),
            'correct_to_wrong': int(((base == 1) & ~correct).sum()),
            'errors_within_group': int((~correct & (coarse[pred] == coarse[y])).sum()),
            'errors_across_groups': int((coarse[pred] != coarse[y]).sum()),
            'mean_target_score': float(true.mean()), 'mean_rival_score': float(rival_score.mean()),
            'mean_signed_margin': float(margin.mean()),
            'mean_target_score_delta': float((true-stats['baseline'][2]).mean()),
            'mean_rival_score_delta': float((rival_score-stats['baseline'][4]).mean()),
            'rival_identity_changed_percent': 100 * float((rival != stats['baseline'][3]).mean())}
        for c in classes:
            mask = y == c['class_id']
            if mask.any():
                per_class.append({'condition': name, 'class_id': c['class_id'], 'class_name': c['class_name'],
                    'coarse_name': c['coarse_name'], 'support': int(mask.sum()),
                    'accuracy_percent': 100 * float(correct[mask].mean()),
                    'delta_vs_baseline_pp': 100 * float(delta[mask].mean()),
                    'mean_target_score_delta': float((true-stats['baseline'][2])[mask].mean()),
                    'mean_rival_score_delta': float((rival_score-stats['baseline'][4])[mask].mean()),
                    'mean_margin_delta': float((margin-stats['baseline'][5])[mask].mean())})
    shuffled = np.mean([stats[name][1].astype(float) for name in stats if name.startswith('shuffle_')], axis=0)
    comparisons = {}
    for reference, values in [('baseline', base), ('neutral', stats['neutral'][1].astype(float)), ('mean_shuffle', shuffled)]:
        delta = stats['category'][1].astype(float) - values
        comparisons[f'category_minus_{reference}'] = {'delta_pp': 100 * float(delta.mean()),
            'ci95_pp': paired_class_ci(delta, y, cfg['bootstrap_draws'], cfg['bootstrap_seed'])}
    return {'conditions': conditions, 'planned_comparisons': comparisons,
            'shuffle_mean_accuracy_percent': 100 * float(shuffled.mean())}, per_class


def run(smoke=False, run_name=None):
    reference = ROOT / 'results/full_cifar100_category_v1'
    inputs = [reference / name for name in ['scores.npz', 'classes.json', 'samples.json', 'manifest.json']]
    out, cfg, manifest = begin('category', smoke, inputs, run_name=run_name)
    engine = Engine(out, cfg, manifest)
    bundle = load_dataset(ROOT, engine.cfg, engine.adapter.preprocess)
    classes = read_json(reference / 'classes.json')
    if bundle.classes != [r['class_name'] for r in classes]:
        raise ValueError('Class order changed.')
    n = 200 if smoke else len(bundle.images)
    from torch.utils.data import Subset
    samples = read_json(reference / 'samples.json')[:n]
    labels = [r['target_id'] for r in samples]
    if labels != bundle.targets[:n] or [r['sample_id'] for r in samples] != bundle.sample_ids[:n]:
        raise ValueError('Dataset sample identity changed.')
    prompts, token_info, permutations = category_prompts(classes, cfg, engine.adapter.tokenizer)
    # Persist all texts and mappings before any new image predictions.
    write_json(out / 'prompts.json', prompts)
    write_json(out / 'neutral_token_matching.json', token_info)
    write_json(out / 'shuffle_permutations.json', permutations)
    write_json(out / 'classes.json', classes)
    write_json(out / 'samples.json', samples)
    features = engine.images(Subset(bundle.images, range(n)),
                             {'dataset': bundle.metadata, 'samples': samples})
    scores = {}
    for name, texts in prompts.items():
        scores[name] = engine.scores(features, engine.texts(texts, batch=100))
        print(f'{name}: scored {n} images against 100 classes', flush=True)
    parity = {}
    with np.load(reference / 'scores.npz', allow_pickle=False) as old:
        for name in ('baseline', 'category'):
            maximum = float(np.max(np.abs(scores[name]-old[name][:n])))
            match = bool(np.array_equal(scores[name].argmax(1), old[name][:n].argmax(1)))
            if not match or maximum > 1e-6:
                raise ValueError(f'Historical parity failed: {name}, max difference={maximum}')
            parity[name] = {'predictions_match': match, 'max_score_difference': maximum, 'n': n}
    np.savez_compressed(out / 'scores.npz', **scores)
    manifest.update(parity=parity, num_samples=n, candidate_count=100, num_conditions=len(scores),
                    dataset=bundle.metadata)
    metrics, per_class = compute_metrics(scores, labels, classes, cfg)
    metrics['neutral_token_length_difference_counts'] = {str(d): sum(r['difference'] == d for r in token_info)
                                                       for d in sorted({r['difference'] for r in token_info})}
    write_json(out / 'metrics.json', metrics)
    write_csv(out / 'per_class.csv', per_class)
    records = []
    for i, sample in enumerate(samples):
        records.append({**sample, **{f'{name}_prediction': int(matrix[i].argmax()) for name, matrix in scores.items()}})
    write_csv(out / 'predictions.csv', records)
    diagnostic_rows, shifts, class_shifts = score_diagnostics(scores, labels, classes)
    write_csv(out / 'category_sample_scores.csv', diagnostic_rows)
    write_json(out / 'category_score_shifts.json', shifts)
    finish(out, manifest)
    print({k: round(v['accuracy_percent'], 4) for k, v in metrics['conditions'].items()}, flush=True)
