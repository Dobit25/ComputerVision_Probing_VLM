"""WS3 (1124.md, Chang 2): offline deep diagnostics for counting, negation and Vietnamese prompts.

    run    compute from committed artifacts only (no inference) into results/final_diagnostics_v1/
    audit  recompute, compare with the saved outputs, and cross-check the T0 evidence matrix

Negation: accuracy by true label (affirmative 'present' vs negative 'absent') overall and per template,
2x2 confusion, affirmative prediction bias, four-trial consistency.
Counting: all vs control-correct vs control-wrong, per true count 1-6 (accuracy, MAE, signed error,
rate of answering 'one'), size regimes, 6x6 confusion.
Vietnamese: per-class accuracy in six prompt conditions, best/zero classes, a score-blind cognate
heuristic, and Spearman correlations between BPE token counts and the per-class accuracy drop.
CIs: group bootstrap (scene group or class), 2,000 draws, seed 1402.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .depth.common import ROOT, cluster_ci, csv_rows, read_json, write_csv
from .io_utils import sha256, write_json

OUT = ROOT / 'results/final_diagnostics_v1'
PROBING = ROOT / 'results/full_probing_clip_b32_v1/predictions.jsonl'
PROBING_METRICS = ROOT / 'results/full_probing_clip_b32_v1/metrics.json'
EVIDENCE = ROOT / 'results/evidence_matrix_1117_v1/metrics.json'
VI = ROOT / 'results/full_vietnamese_cifar100_v1'
CIFAR_BASE = ROOT / 'results/full_cifar100_clip_b32_v1/predictions.csv'
INPUTS = [PROBING, PROBING_METRICS, EVIDENCE, VI / 'scores.npz', VI / 'metrics.json', VI / 'bpe_diagnostics.json',
          VI / 'multilingual_classes.json', CIFAR_BASE]
DRAWS, SEED = 2000, 1402
VI_CONDITIONS = ('en_baseline', 'vi_diacritics', 'vi_no_diacritics', 'en_frame_vi_label', 'vi_frame_en_label',
                 'vi_en_bilingual')
COGNATE_MIN_RATIO = 0.5  # fixed before looking at per-class accuracy


# ---------------------------------------------------------------- helpers

def rate(flags, groups):
    flags = np.asarray(flags, dtype=float)
    return {'n': int(flags.size), 'k': int(flags.sum()), 'percent': float(100 * flags.mean()),
            'ci95_percent': cluster_ci(flags, groups, DRAWS, SEED)}


def rankdata(x):
    """Average ranks (ties share the mean rank), 1-based."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tie = x == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ci(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = [spearman(x[i], y[i]) for i in rng.integers(len(x), size=(DRAWS, len(x)))]
    return np.nanquantile(draws, [.025, .975]).tolist()


def load_probing():
    with PROBING.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- negation

def negation_metrics(rows):
    neg = [r for r in rows if r['task'] == 'negation']
    for r in neg:
        if r['target'] != (0 if r['stratum'] == 'present' else 1):
            raise ValueError('Unexpected negation label encoding.')
    out, table = {}, []
    for template in ('all', 'with_without', 'contains_does_not'):
        sub = [r for r in neg if template == 'all' or r['condition'] == template]
        groups = [r['group_id'] for r in sub]
        entry = {'accuracy': rate([r['correct'] for r in sub], groups),
                 'affirmative_prediction_rate': rate([r['prediction'] == 0 for r in sub], groups)}
        confusion = {}
        for true in ('present', 'absent'):
            part = [r for r in sub if r['stratum'] == true]
            entry[f'accuracy_{true}'] = rate([r['correct'] for r in part], [r['group_id'] for r in part])
            confusion[true] = {'pred_present': sum(r['prediction'] == 0 for r in part),
                               'pred_absent': sum(r['prediction'] == 1 for r in part)}
            table.append({'template': template, 'true_label': true, 'n': len(part),
                          'correct': sum(r['correct'] for r in part),
                          'accuracy_percent': round(entry[f'accuracy_{true}']['percent'], 4),
                          'pred_present': confusion[true]['pred_present'],
                          'pred_absent': confusion[true]['pred_absent']})
        entry['confusion_true_x_pred'] = confusion
        entry['accuracy_gap_absent_minus_present_pp'] = (entry['accuracy_absent']['percent']
                                                         - entry['accuracy_present']['percent'])
        out[template] = entry
    by_group = {}
    for r in neg:
        by_group.setdefault(r['group_id'], []).append(r['correct'])
    if any(len(v) != 4 for v in by_group.values()):
        raise ValueError('Each negation group must have four trials.')
    out['four_trial_consistency'] = {'groups': len(by_group), 'all_correct': sum(all(v) for v in by_group.values()),
                                     'percent': float(100 * np.mean([all(v) for v in by_group.values()]))}
    out['n_trials'] = len(neg)
    return out, table


# ---------------------------------------------------------------- counting

def counting_metrics(rows):
    cnt = [r for r in rows if r['task'] == 'counting']
    if any(r['target'] != r['number'] - 1 for r in cnt):
        raise ValueError('Unexpected counting label encoding.')
    def block(sub):
        groups = [r['group_id'] for r in sub]
        err = np.array([(r['prediction'] + 1) - r['number'] for r in sub], dtype=float)
        return {'accuracy': rate([r['correct'] for r in sub], groups),
                'mae': float(np.abs(err).mean()), 'mean_signed_error': float(err.mean()),
                'answer_one_rate_percent': float(100 * np.mean([r['prediction'] == 0 for r in sub]))}
    out = {'n_images': len(cnt), 'all': block(cnt),
           'control_correct': block([r for r in cnt if r['control_correct']]),
           'control_wrong': block([r for r in cnt if not r['control_correct']]),
           'by_condition': {c: block([r for r in cnt if r['condition'] == c]) for c in ('fixed_radius', 'fixed_total_area')}}
    table, per_count = [], {}
    for n in range(1, 7):
        sub = [r for r in cnt if r['number'] == n]
        cc = [r for r in sub if r['control_correct']]
        per_count[str(n)] = {'all': block(sub), 'control_correct': block(cc) if cc else None}
        for scope, part in (('all', sub), ('control_correct', cc)):
            b = block(part)
            table.append({'true_count': n, 'scope': scope, 'n': len(part), 'correct': b['accuracy']['k'],
                          'accuracy_percent': round(b['accuracy']['percent'], 4), 'mae': round(b['mae'], 4),
                          'mean_signed_error': round(b['mean_signed_error'], 4),
                          'answer_one_rate_percent': round(b['answer_one_rate_percent'], 4)})
    out['by_true_count'] = per_count
    confusion = np.zeros((6, 6), dtype=int)
    for r in cnt:
        confusion[r['number'] - 1, r['prediction']] += 1
    out['confusion_rows_true_cols_pred'] = confusion.tolist()
    return out, table


# ---------------------------------------------------------------- Vietnamese

def cognate_ratio(en_label, vi_plain):
    en = en_label.replace('_', ' ').lower()
    vi = vi_plain.lower()
    return 1.0 if en in vi else difflib.SequenceMatcher(None, en, vi).ratio()


def vietnamese_metrics():
    base = csv_rows(CIFAR_BASE)
    targets = np.array([int(r['target_id']) for r in sorted(base, key=lambda r: int(r['test_index']))])
    baseline_pred = np.array([int(r['prediction_id']) for r in sorted(base, key=lambda r: int(r['test_index']))])
    with np.load(VI / 'scores.npz', allow_pickle=False) as z:
        scores = {k: z[k] for k in z.files}
    if not np.array_equal(scores['en_baseline'].argmax(1), baseline_pred):
        raise ValueError('Vietnamese en_baseline scores are not aligned with the CIFAR-100 baseline predictions.')
    classes = read_json(VI / 'multilingual_classes.json')
    bpe = {r['class_id']: r for r in read_json(VI / 'bpe_diagnostics.json')['per_class_bpe']}
    stored = read_json(VI / 'metrics.json')['single_templates']
    correct = {c: scores[c].argmax(1) == targets for c in VI_CONDITIONS}
    overall = {c: float(100 * correct[c].mean()) for c in VI_CONDITIONS}
    for c in VI_CONDITIONS:
        if abs(overall[c] - stored[c]['accuracy_percent']) > 1e-9:
            raise ValueError(f'{c}: recomputed accuracy differs from the saved Vietnamese metrics.')
    overall['en_vi_bilingual_embedding_ensemble'] = float(
        100 * (scores['en_vi_bilingual_embedding_ensemble'].argmax(1) == targets).mean())
    per_class = []
    for cls in classes:
        cid = cls['class_id']
        mask = targets == cid
        row = {'class_id': cid, 'en_label': cls['en_label'], 'vi_label': cls['vi_label'],
               'vi_no_diacritics_label': cls['vi_no_diacritics_label'], 'n_images': int(mask.sum()),
               'en_label_tokens': bpe[cid]['en_baseline_label_tokens'],
               'vi_label_tokens': bpe[cid]['vi_diacritics_label_tokens'],
               'vi_no_diacritics_label_tokens': bpe[cid]['vi_no_diacritics_label_tokens']}
        for c in VI_CONDITIONS:
            row[f'acc_{c}'] = round(float(100 * correct[c][mask].mean()), 4)
        row['drop_vi_diacritics_pp'] = round(row['acc_en_baseline'] - row['acc_vi_diacritics'], 4)
        row['drop_vi_no_diacritics_pp'] = round(row['acc_en_baseline'] - row['acc_vi_no_diacritics'], 4)
        row['cognate_ratio'] = round(cognate_ratio(cls['en_label'], cls['vi_no_diacritics_label']), 4)
        row['cognate_like'] = row['cognate_ratio'] >= COGNATE_MIN_RATIO
        per_class.append(row)
    col = lambda k: np.array([r[k] for r in per_class], dtype=float)
    ranked = sorted(per_class, key=lambda r: (-r['acc_vi_diacritics'], r['class_id']))
    zero = [r for r in per_class if r['acc_vi_diacritics'] == 0]
    cog = [r for r in per_class if r['cognate_like']]
    nat = [r for r in per_class if not r['cognate_like']]
    corr = {}
    for name, x, y in (('vi_label_tokens_vs_drop_vi', col('vi_label_tokens'), col('drop_vi_diacritics_pp')),
                       ('vi_no_diacritics_tokens_vs_drop_vi_no_diacritics', col('vi_no_diacritics_label_tokens'),
                        col('drop_vi_no_diacritics_pp')),
                       ('vi_over_en_token_ratio_vs_drop_vi', col('vi_label_tokens') / col('en_label_tokens'),
                        col('drop_vi_diacritics_pp')),
                       ('vi_label_tokens_vs_acc_vi', col('vi_label_tokens'), col('acc_vi_diacritics'))):
        corr[name] = {'spearman_rho': spearman(x, y), 'ci95_class_bootstrap': spearman_ci(x, y), 'n_classes': len(x)}
    metrics = {
        'overall_top1_percent': overall,
        'top10_vi_diacritics_classes': [{k: r[k] for k in ('class_id', 'en_label', 'vi_label', 'acc_vi_diacritics',
                                                          'acc_en_baseline')} for r in ranked[:10]],
        'zero_accuracy_vi_diacritics': {'n_classes': len(zero), 'classes': [r['en_label'] for r in zero]},
        'cognate_heuristic': {'rule': f'en label in vi label without diacritics, or difflib ratio >= {COGNATE_MIN_RATIO}',
                              'n_cognate_like': len(cog), 'n_other': len(nat),
                              'mean_acc_vi_diacritics_cognate_like': float(np.mean([r['acc_vi_diacritics'] for r in cog])) if cog else None,
                              'mean_acc_vi_diacritics_other': float(np.mean([r['acc_vi_diacritics'] for r in nat])) if nat else None,
                              'cognate_like_classes': [f"{r['en_label']} = {r['vi_label']}" for r in cog]},
        'spearman': corr,
        'mean_label_tokens': {'en': float(col('en_label_tokens').mean()), 'vi': float(col('vi_label_tokens').mean()),
                              'vi_no_diacritics': float(col('vi_no_diacritics_label_tokens').mean())},
        'n_images': int(targets.size),
    }
    return metrics, per_class


# ---------------------------------------------------------------- cross-checks

def evidence_checks(counting, negation):
    ev = read_json(EVIDENCE)
    checks = {
        'counting_confusion': counting['confusion_rows_true_cols_pred'] == ev['counting']['confusion_rows_true_cols_pred'],
        'counting_correct': counting['all']['accuracy']['k'] == ev['counting']['correct'],
        'counting_mae': abs(counting['all']['mae'] - ev['counting']['mae']) < 1e-12,
        'counting_signed': abs(counting['all']['mean_signed_error'] - ev['counting']['mean_signed_error']) < 1e-12,
        'counting_conditions': all(counting['by_condition'][c]['accuracy']['k'] == v['correct']
                                   for c, v in ev['counting']['groups']['condition'].items()),
        'negation_correct': negation['all']['accuracy']['k'] == ev['negation']['correct'],
        'negation_templates': all(negation[c]['accuracy']['k'] == v['correct']
                                  for c, v in ev['negation']['groups']['condition'].items()),
        'negation_strata': all(negation['all'][f'accuracy_{s}']['k'] == v['correct']
                               for s, v in ev['negation']['groups']['stratum'].items()),
    }
    core = read_json(PROBING_METRICS)
    checks['counting_control_correct'] = counting['control_correct']['accuracy']['n'] == core['counting']['control_correct_samples']
    checks['negation_consistency'] = abs(negation['four_trial_consistency']['percent'] / 100
                                         - core['negation']['all_rows_correct_group_rate']) < 1e-12
    checks['negation_affirmative_rate'] = abs(negation['all']['affirmative_prediction_rate']['percent'] / 100
                                              - core['negation']['affirmative_choice_rate']) < 1e-12
    if not all(checks.values()):
        raise ValueError(f'Cross-check with evidence_matrix_1117_v1 failed: {checks}')
    reference = {'counting_seed0_ci95_percent': [100 * v for v in core['counting']['accuracy_ci95_group_bootstrap']],
                 'negation_seed0_ci95_percent': [100 * v for v in core['negation']['accuracy_ci95_group_bootstrap']]}
    return checks, reference


def compute():
    rows = load_probing()
    negation, neg_table = negation_metrics(rows)
    counting, count_table = counting_metrics(rows)
    vietnamese, per_class = vietnamese_metrics()
    checks, reference = evidence_checks(counting, negation)
    metrics = {'study': 'final_diagnostics_v1', 'new_inference': False,
               'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'scene group (probes) or class (Spearman)'},
               'negation': negation, 'counting': counting, 'vietnamese': vietnamese,
               'cross_checks_evidence_matrix_1117': checks, 'reference_original_seed0': reference}
    return metrics, {'negation_deep.csv': neg_table, 'counting_deep.csv': count_table,
                     'vietnamese_per_class.csv': per_class}


def run():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; results are immutable (delete it explicitly to regenerate).')
    metrics, tables = compute()
    OUT.mkdir(parents=True)
    write_json(OUT / 'metrics.json', metrics)
    for name, rows in tables.items():
        write_csv(OUT / name, rows)
    manifest = {'study': 'final_diagnostics_v1', 'status': 'completed', 'new_inference': False,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'code_sha256': sha256(Path(__file__)),
                'input_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in INPUTS},
                'artifact_sha256': {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()}}
    write_json(OUT / 'manifest.json', manifest)
    print(json.dumps(metrics['cross_checks_evidence_matrix_1117'], indent=1), flush=True)


def audit():
    from .extension_artifacts import close_enough
    m = read_json(OUT / 'manifest.json')
    for name, digest in m['input_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Changed input: {name}')
    for name, digest in m['artifact_sha256'].items():
        if sha256(OUT / name) != digest:
            raise ValueError(f'Changed artifact: {name}')
    metrics, tables = compute()
    close_enough(metrics, read_json(OUT / 'metrics.json'))
    for name, rows in tables.items():
        if [{k: str(v) for k, v in r.items()} for r in rows] != csv_rows(OUT / name):
            raise ValueError(f'{name} differs from recomputation.')
    result = {'verified': True, 'new_inference': False, 'cross_checks': metrics['cross_checks_evidence_matrix_1117'],
              'denominators': {'counting': metrics['counting']['n_images'], 'negation': metrics['negation']['n_trials'],
                               'vietnamese': metrics['vietnamese']['n_images']}}
    write_json(OUT.parent / (OUT.name + '_audit.json'), result)
    print(json.dumps(result, indent=1), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['run', 'audit'])
    a = p.parse_args(argv)
    run() if a.action == 'run' else audit()


if __name__ == '__main__':
    main()
