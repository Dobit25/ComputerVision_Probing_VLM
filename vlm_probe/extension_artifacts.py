"""Auditable extension artifacts and non-mutating legacy-run review (NumPy only)."""
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from .depth.common import ROOT, read_json, write_csv, finish
from .io_utils import sha256, source_snapshot, write_json
from .prompt_study.metrics import (summarize_template_suite, logit_average_ensemble,
                                  probability_average_ensemble)

PROTOCOL = ROOT / 'docs/integration_protocol_v2.md'
REVIEW = ROOT / 'results/review_hieu_v2'


def begin_extension(stage, config_path, inputs=(), smoke=False, run_name=None):
    from .depth.common import begin
    out, cfg, manifest = begin(stage, smoke, inputs=[config_path, PROTOCOL, *inputs], run_name=run_name)
    # Preserve the model/evaluation defaults separately; bind the actual study design.
    write_json(out / 'evaluation_config.json', cfg)
    write_json(out / 'study_config.json', read_json(config_path))
    (out / 'protocol.md').write_bytes(PROTOCOL.read_bytes())
    extra = [config_path, PROTOCOL, ROOT / 'configs/probing_depth_v1.json',
             ROOT / 'configs/clip_b32_cifar100.json', *inputs]
    manifest['source_sha256'], manifest['source_snapshot'] = source_snapshot(ROOT, extra)
    manifest['artifact_schema'] = 'extensions_review_v2'
    write_json(out / 'manifest.json', manifest)
    return out, cfg, manifest


def baseline_rows(run):
    with (ROOT / run / 'predictions.csv').open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def expanded_scores(scores, scale=100.0):
    singles = {k: v for k, v in scores.items() if v.ndim == 2 and
               k not in ('text_embedding_ensemble', 'en_vi_bilingual_embedding_ensemble')}
    result = {**singles,
              'logit_average_ensemble': logit_average_ensemble(list(singles.values())),
              'probability_average_ensemble': probability_average_ensemble(list(singles.values()), scale)}
    result.update({k: v for k, v in scores.items() if k in
                   ('text_embedding_ensemble', 'en_vi_bilingual_embedding_ensemble')})
    return singles, result


def prediction_tables(scores, samples, classes, baseline_key, scale=100.0):
    _, conditions = expanded_scores(scores, scale)
    y = np.asarray([r['target_id'] for r in samples])
    base = conditions[baseline_key].argmax(1)
    rows, per_class, diagnostics = [], [], {}
    for key, mat in conditions.items():
        if mat.shape != (len(samples), len(classes)) or not np.isfinite(mat).all():
            raise ValueError(f'Invalid scores: {key}')
        pred = mat.argmax(1); correct = pred == y
        for i, sample in enumerate(samples):
            rows.append({'sample_id': sample['sample_id'], 'condition': key,
                         'target_id': int(y[i]), 'prediction_id': int(pred[i]),
                         'correct': bool(correct[i]), 'target_score': float(mat[i, y[i]])})
        accs = []
        for c, name in enumerate(classes):
            mask = y == c; support = int(mask.sum())
            acc = float(correct[mask].mean() * 100) if support else None
            if support: accs.append(acc)
            per_class.append({'condition': key, 'class_id': c, 'class_name': name,
                              'support': support, 'correct': int(correct[mask].sum()), 'accuracy_percent': acc})
        diagnostics[key] = {'macro_accuracy_percent': float(np.mean(accs)),
                            'prediction_changed': int((pred != base).sum()),
                            'wrong_to_correct': int(((base != y) & correct).sum()),
                            'correct_to_wrong': int(((base == y) & ~correct).sum())}
    return rows, per_class, diagnostics


def save_classification(out, scores, samples, classes, baseline_key, scale=100.0):
    write_json(out / 'samples.json', samples); write_json(out / 'classes.json', classes)
    rows, per_class, diag = prediction_tables(scores, samples, classes, baseline_key, scale)
    write_csv(out / 'predictions.csv', rows); write_csv(out / 'per_class.csv', per_class)
    write_json(out / 'diagnostics.json', diag)


def verify_manifest(directory):
    m = read_json(directory / 'manifest.json')
    if m['status'] != 'completed': raise ValueError('Incomplete run')
    files = {p.name for p in directory.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if files != set(m['artifact_sha256']): raise ValueError('Artifact inventory differs')
    for name, h in m['artifact_sha256'].items():
        if sha256(directory / name) != h: raise ValueError(f'Artifact hash: {name}')
    snap = m['source_snapshot']
    if sha256(ROOT / snap['path']) != snap['sha256']: raise ValueError('Snapshot hash')
    with zipfile.ZipFile(ROOT / snap['path']) as z:
        if set(z.namelist()) != set(m['source_sha256']): raise ValueError('Snapshot inventory')
        for name, h in m['source_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != h: raise ValueError('Archived source hash')
        for name, h in m.get('input_sha256', {}).items():
            # A later amendment may change the working protocol. Audit the bytes
            # actually frozen with this run; data not archived must still match.
            if name in z.namelist():
                if hashlib.sha256(z.read(name)).hexdigest() != h: raise ValueError(f'Archived input hash: {name}')
            elif sha256(ROOT / name) != h: raise ValueError(f'Input hash: {name}')
    return m


def close_enough(actual, expected, path='metrics'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise ValueError(f'{path}: keys')
        for k in expected: close_enough(actual[k], expected[k], path + '/' + k)
    elif isinstance(expected, list):
        if len(actual) != len(expected): raise ValueError(path)
        for a, e in zip(actual, expected): close_enough(a, e, path)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-9): raise ValueError(path)
    elif actual != expected: raise ValueError(path)


def audit_classification(directory, manifest, output):
    vi = manifest['stage'] == 'vietnamese_cifar100'
    dataset = 'cifar100' if vi else manifest['dataset']
    # Legacy snapshot did not include extension configs: validate pinned input bytes.
    config_path = ROOT / ('configs/vietnamese_study_v1.json' if vi else 'configs/prompt_study_v1.json')
    cfg = read_json(directory / 'study_config.json') if (directory / 'study_config.json').exists() else read_json(config_path)
    if not (directory / 'study_config.json').exists():
        if sha256(config_path) != manifest['input_sha256'][config_path.relative_to(ROOT).as_posix()]:
            raise ValueError('Legacy config input differs')
    from .prompt_study.design import load_study_config, build_dataset_templates
    run = load_study_config()['datasets'][dataset]['baseline_run']
    base = baseline_rows(run)
    n = manifest['evaluated_samples']
    expected_n = min(200, len(base)) if manifest['mode'] == 'smoke' else len(base)
    if n != expected_n: raise ValueError('Wrong sample count for full/smoke mode')
    base = base[:n]
    samples = [{'sample_id': r['sample_id'], 'target_id': int(r['target_id'])} for r in base]
    classes_raw = read_json(ROOT / run / 'classes.json')
    classes = [r['dataset_name'] if isinstance(r, dict) else r for r in classes_raw]
    if classes and isinstance(classes[0], dict):
        raise ValueError('Unrecognized baseline classes schema')
    if (directory / 'samples.json').exists(): close_enough(read_json(directory / 'samples.json'), samples)
    if (directory / 'classes.json').exists(): close_enough(read_json(directory / 'classes.json'), classes)
    with np.load(directory / 'scores.npz', allow_pickle=False) as z: scores = dict(z)
    key = 'en_baseline' if vi else 'T0_baseline'
    if not vi:
        close_enough(read_json(directory / 'templates.json'), build_dataset_templates(dataset, classes, cfg))
    else:
        from .vietnamese_study.lexicon import build_cifar100_multilingual_prompts
        _, records = build_cifar100_multilingual_prompts(classes)
        close_enough(read_json(directory / 'multilingual_classes.json'), records)
    singles, _ = expanded_scores(scores)
    y = np.asarray([r['target_id'] for r in samples])
    extras = {k: v for k, v in scores.items() if k in
              ('text_embedding_ensemble', 'en_vi_bilingual_embedding_ensemble')}
    computed = summarize_template_suite(singles, y, key, extras,
        logit_scale=cfg.get('clip_logit_scale', 100.0),
        bootstrap_draws=cfg['bootstrap_draws'], bootstrap_seed=cfg['bootstrap_seed'])
    stored = read_json(directory / 'metrics.json')
    for name, value in computed.items(): close_enough(stored[name], value, name)
    ensemble_check = False
    if not vi and 'class_alignment_norms' in scores:
        derived = np.mean(np.stack(list(singles.values())).astype(np.float64), axis=0) / scores['class_alignment_norms']
        if not np.allclose(derived, scores['text_embedding_ensemble'], atol=1e-6, rtol=0):
            raise ValueError('Embedding ensemble differs from normalized template average')
        ensemble_check = True
    if (directory / 'text_embeddings.npz').exists():
        from .prompt_study.metrics import text_embedding_ensemble
        with np.load(directory / 'text_embeddings.npz', allow_pickle=False) as z: embeddings = dict(z)
        selected_keys = ['en_baseline','vi_diacritics','vi_en_bilingual'] if vi else list(singles)
        _, norms = text_embedding_ensemble([embeddings[k] for k in selected_keys])
        derived = np.mean(np.stack([singles[k] for k in selected_keys]).astype(np.float64), axis=0) / norms
        ensemble_key = 'en_vi_bilingual_embedding_ensemble' if vi else 'text_embedding_ensemble'
        if not np.allclose(derived, scores[ensemble_key], atol=1e-6, rtol=0): raise ValueError('Embedding ensemble identity')
        ensemble_check = True
    pred = scores[key].argmax(1)
    if not np.array_equal(pred, [int(r['prediction_id']) for r in base]): raise ValueError('Baseline prediction parity')
    true_scores = scores[key][np.arange(n), y]
    maxdiff = float(np.max(abs(true_scores - np.asarray([float(r['target_cosine']) for r in base]))))
    if maxdiff > 1e-6: raise ValueError(f'Baseline cosine parity {maxdiff}')
    output.mkdir(parents=True, exist_ok=True)
    save_classification(output, scores, samples, classes, key, cfg.get('clip_logit_scale', 100.0))
    if (directory / 'predictions.csv').exists():
        if (directory / 'predictions.csv').read_bytes() != (output / 'predictions.csv').read_bytes():
            raise ValueError('Stored predictions differ')
    if (directory / 'per_class.csv').exists():
        if (directory / 'per_class.csv').read_bytes() != (output / 'per_class.csv').read_bytes():
            raise ValueError('Stored class metrics differ')
    return {'metrics_recomputed': True, 'n': n, 'baseline_predictions_identical': True,
            'baseline_target_cosine_max_abs_difference': maxdiff,
            'text_embedding_ensemble_construction_recomputed': ensemble_check,
            'limitation': None if ensemble_check else 'Stored embedding-ensemble scores are audited; legacy raw text vectors were not saved.'}


def audit_run(name):
    directory = ROOT / 'results' / name
    if directory.parent != ROOT / 'results': raise ValueError('Simple run name required')
    manifest = verify_manifest(directory)
    output = REVIEW / name
    output.mkdir(parents=True, exist_ok=True)
    stage = manifest['stage']
    if stage.startswith('prompt_') and stage != 'prompt_probing' or stage == 'vietnamese_cifar100':
        details = audit_classification(directory, manifest, output)
    elif stage == 'spatial_mitigation':
        from .depth.design import spatial_images
        rows = spatial_images(read_json(ROOT / 'configs/probing_depth_v1.json'), materialize=False)
        rows = rows[:manifest['num_images']]
        with np.load(directory / 'relational_scores.npz', allow_pickle=False) as z: scores = z['scores']
        y = np.asarray([r['target'] for r in rows]); pred = scores.argmax(1)
        if scores.shape != (len(rows), 4) or not np.isfinite(scores).all(): raise ValueError('Invalid final scores')
        import collections
        grouped = collections.defaultdict(list)
        for r, p in zip(rows, pred): grouped[(r['group_id'], r['variant'])].append((r['target'], int(p)))
        stats = read_json(directory / 'metrics.json')
        close_enough(stats['four_way_accuracy_percent'], 100 * float((pred == y).mean()))
        from .depth.common import cluster_ci, csv_rows
        cfg = read_json(ROOT / 'configs/spatial_diagnostics_v1.json')
        close_enough(stats['four_way_ci95_percent'], cluster_ci(pred == y, [r['group_id'] for r in rows], cfg['bootstrap_draws'], cfg['bootstrap_seed']))
        axis_pred = np.array([int(scores[i, :2].argmax()) if target < 2 else int(scores[i, 2:].argmax()) + 2 for i, target in enumerate(y)])
        close_enough(stats['within_axis_two_way_accuracy_percent'], 100 * float((axis_pred == y).mean()))
        stored_predictions = csv_rows(directory / 'predictions.csv')
        if len(stored_predictions) != len(rows): raise ValueError('Prediction count')
        for i, (r, stored_row) in enumerate(zip(rows, stored_predictions)):
            if stored_row['image_id'] != r['image_id'] or int(stored_row['target']) != y[i] or int(stored_row['prediction']) != pred[i]:
                raise ValueError('Prediction identity or argmax')
            if stored_row['correct'] != str(bool(pred[i] == y[i])): raise ValueError('Correctness flag')
        for axis, field in [((0, 1), 'horizontal_pair_both_correct_percent'),
                            ((2, 3), 'vertical_pair_both_correct_percent'),
                            ((0, 1, 2, 3), 'all_four_relations_correct_percent')]:
            close_enough(stats[field], 100 * float(np.mean([all(t == p for t, p in g if t in axis) for g in grouped.values()])))
        details = {'metrics_recomputed': True, 'n': len(rows),
                   'region_to_relational_scores_recomputed': False,
                   'limitation': 'Legacy run saved final relational scores, not per-region scores.'}
    elif stage == 'prompt_probing':
        if not (directory / 'counting_scores.npz').exists():
            details = {'metrics_recomputed': False, 'artifact_integrity_verified': True,
                       'limitation': 'Legacy aggregate-only run; requires a new inference run to recover per-sample scores.'}
        else:
            cfg = read_json(directory / 'study_config.json'); stored = read_json(directory / 'metrics.json')
            for task in ['counting', 'negation']:
                rows = read_json(directory / f'{task}_samples.json')
                frozen = read_json(ROOT / 'results/full_probing_clip_b32_v1/samples.json')
                expected = [r for r in frozen if r['task'] == task and
                            (task == 'counting' or r['condition'] == 'with_without')]
                expected_n = min(60,len(expected)) if manifest['mode']=='smoke' else len(expected)
                if len(rows) != expected_n: raise ValueError('Wrong probing sample count')
                expected = expected[:expected_n]
                close_enough(rows, expected, 'probe_samples')
                from .prompt_study.design import build_counting_prompts, build_negation_prompts
                templates = [(build_counting_prompts(r['color'], r['shape'], cfg) if task == 'counting' else
                              build_negation_prompts(r['target_color'], r['target_shape'], cfg)) for r in rows]
                close_enough(read_json(directory / f'{task}_templates.json'), templates, 'captions')
                with np.load(directory / f'{task}_scores.npz', allow_pickle=False) as z: scores = dict(z)
                singles, conditions = expanded_scores(scores)
                computed = summarize_template_suite(singles, [r['target'] for r in rows], next(iter(singles)),
                    {'text_embedding_ensemble': scores['text_embedding_ensemble']},
                    groups=[r['group_id'] for r in rows], logit_scale=cfg['clip_logit_scale'],
                    bootstrap_draws=cfg['bootstrap_draws'], bootstrap_seed=cfg['bootstrap_seed'])
                close_enough(computed, stored[task])
                from .depth.common import csv_rows
                expected_predictions = [{'sample_id': r['sample_id'], 'group_id': r['group_id'],
                    'condition': name, 'target': str(r['target']), 'prediction': str(int(mat[i].argmax()))}
                    for name, mat in conditions.items() for i, r in enumerate(rows)]
                if expected_predictions != csv_rows(directory / f'{task}_predictions.csv'):
                    raise ValueError('Probing predictions differ')
            details = {'metrics_recomputed': True}
    elif stage.startswith('natural_region_'):
        from .spatial_diagnostics.natural_regions import audit as natural_audit
        details = natural_audit(directory)
    else: raise ValueError(f'Unsupported extension stage: {stage}')
    payload = {'run': directory.relative_to(ROOT).as_posix(), 'artifact_integrity_verified': True,
               'manifest_sha256': sha256(directory / 'manifest.json'), **details}
    payload['legacy_config_protocol_note'] = (None if manifest.get('artifact_schema') else
        'Legacy config.json/protocol.md contain depth defaults; extension configs/protocols are identified by input_sha256. New runs use study_config.json and the actual protocol.')
    payload['derived_sha256'] = {p.name: sha256(p) for p in output.iterdir() if p.name != 'audit.json' and p.is_file()}
    write_json(output / 'audit.json', payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run_name')
    audit_run(parser.parse_args().run_name)
