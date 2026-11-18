"""Recompute decisions/metrics and verify inputs, design, and source snapshots."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from ..io_utils import sha256, write_json
from .common import ROOT, read_json, csv_rows


def equal(a, b, path='root'):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError(f'Key mismatch: {path}')
        for k in a:
            equal(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list):
        if len(a) != len(b):
            raise ValueError(f'Length mismatch: {path}')
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f'{path}[{i}]')
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not np.isclose(a, b, atol=1e-10, rtol=1e-9):
            raise ValueError(f'Numeric mismatch: {path}: {a} != {b}')
    elif a != b:
        raise ValueError(f'Value mismatch: {path}')


def verify_hashes(out):
    manifest = read_json(out / 'manifest.json')
    if manifest['status'] != 'completed':
        raise ValueError('Incomplete run.')
    files = {p.name for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if files != set(manifest['artifact_sha256']):
        raise ValueError('Unexpected or missing run artifact.')
    for name, digest in manifest['artifact_sha256'].items():
        if sha256(out / name) != digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    for name, digest in manifest['input_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Input checksum mismatch: {name}')
    snapshot = ROOT / manifest['source_snapshot']['path']
    if sha256(snapshot) != manifest['source_snapshot']['sha256']:
        raise ValueError('Source snapshot checksum mismatch.')
    with zipfile.ZipFile(snapshot) as z:
        if set(z.namelist()) != set(manifest['source_sha256']):
            raise ValueError('Source snapshot file mismatch.')
        for name, digest in manifest['source_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('Snapshot source changed.')
    if sha256(out / 'protocol.md') != manifest['source_sha256']['docs/probing_depth_protocol.md']:
        raise ValueError('Protocol snapshot differs.')
    return manifest


def audit_one(out):
    manifest = verify_hashes(out)
    stage = manifest['stage']
    cfg = read_json(out / 'config.json')
    result = {'verified': True, 'stage': stage, 'mode': manifest['mode'], 'run': out.relative_to(ROOT).as_posix()}
    if stage in ('spatial', 'natural'):
        from .spatial import compute_metrics
        from .design import spatial_images, spatial_trials
        images, trials = read_json(out / 'images.json'), read_json(out / 'trials.json')
        expected = spatial_trials(images, cfg, 'drawing' if stage == 'spatial' else 'photo')
        for r in expected:
            im = images[r['image_index']]
            if 'control_captions' in im:
                r['control_captions'], r['control_target'] = im['control_captions'], im['control_target']
        equal(expected, trials, 'caption_design')
        if stage == 'spatial':
            from ..probing.stimuli import render
            designed = spatial_images(cfg, materialize=False)[:len(images)]
            expected_count = 36 if manifest['mode'] == 'smoke' else 1440
            if len(images) != expected_count:
                raise ValueError('Spatial image count mismatch.')
            for im, design in zip(images, designed):
                equal({k: v for k, v in im.items() if k not in ['image_path', 'image_sha256']}, design, 'scene_design')
                buffer = io.BytesIO()
                render(design).save(buffer, format='PNG')
                if hashlib.sha256(buffer.getvalue()).hexdigest() != im['image_sha256']:
                    raise ValueError('Rendered scene differs from design.')
        else:
            from .natural import validate_selected
            validate_selected(images, cfg)
        for im in images:
            if sha256(ROOT / im['image_path']) != im['image_sha256']:
                raise ValueError('Input image checksum mismatch.')
        with np.load(out / 'scores.npz', allow_pickle=False) as z:
            arrays = dict(z)
        if arrays['scores'].shape != (len(trials), 4) or any(not np.isfinite(x).all() or (abs(x) > 1.00001).any() for x in arrays.values()):
            raise ValueError('Invalid candidate scores.')
        metrics, records = compute_metrics(trials, config=cfg, **arrays)
        equal(metrics, read_json(out / 'metrics.json'), 'metrics')
        stored = csv_rows(out / 'predictions.csv')
        if [{k: str(v) for k, v in r.items()} for r in records] != stored:
            raise ValueError('Saved predictions differ from scores.')
        result.update(num_images=len(images), num_trials=len(trials), metrics_recomputed=True)
    elif stage == 'category':
        from .category import compute_metrics
        from .design import category_prompts
        import open_clip
        classes, samples = read_json(out / 'classes.json'), read_json(out / 'samples.json')
        prompts, tokens, permutations = category_prompts(classes, cfg, open_clip.get_tokenizer('ViT-B-32'))
        equal(prompts, read_json(out / 'prompts.json'), 'prompts')
        equal(tokens, read_json(out / 'neutral_token_matching.json'), 'tokens')
        equal(permutations, read_json(out / 'shuffle_permutations.json'), 'permutations')
        with np.load(out / 'scores.npz', allow_pickle=False) as z:
            scores = dict(z)
        if set(scores) != set(prompts) or len(scores) != 8:
            raise ValueError('Category conditions mismatch.')
        old = ROOT / cfg['historical_category']
        if samples != read_json(old / 'samples.json')[:len(samples)]:
            raise ValueError('Category sample order changed.')
        expected_n = 200 if manifest['mode'] == 'smoke' else 10000
        if len(samples) != expected_n:
            raise ValueError('Category sample count mismatch.')
        with np.load(old / 'scores.npz', allow_pickle=False) as ref:
            for name in ('baseline', 'category'):
                if np.max(abs(scores[name]-ref[name][:len(samples)])) > 1e-6 or not np.array_equal(scores[name].argmax(1), ref[name][:len(samples)].argmax(1)):
                    raise ValueError('Historical baseline/category parity failed.')
        metrics, per_class = compute_metrics(scores, [r['target_id'] for r in samples], classes, cfg)
        metrics['neutral_token_length_difference_counts'] = {str(d): sum(r['difference'] == d for r in tokens) for d in sorted({r['difference'] for r in tokens})}
        equal(metrics, read_json(out / 'metrics.json'), 'metrics')
        if [{k: str(v) for k, v in r.items()} for r in per_class] != csv_rows(out / 'per_class.csv'):
            raise ValueError('Per-class metrics differ.')
        expected = [{**r, **{f'{name}_prediction': int(s[i].argmax()) for name, s in scores.items()}} for i, r in enumerate(samples)]
        if [{k: str(v) for k, v in r.items()} for r in expected] != csv_rows(out / 'predictions.csv'):
            raise ValueError('Predictions differ from scores.')
        result.update(num_samples=len(samples), num_conditions=8, metrics_recomputed=True, historical_parity=True)
    elif stage == 'historical':
        from .analysis import score_diagnostics
        old = ROOT / cfg['historical_probing']
        cat = ROOT / cfg['historical_category']
        records = [json.loads(line) for line in (old / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        stored = read_json(out / 'metrics.json')
        from collections import defaultdict
        for task, stats in stored['probing'].items():
            selected = [r for r in records if r['task'] == task]
            k = len(selected[0]['scores'])
            confusion = np.zeros((k, k), dtype=int)
            grouped = defaultdict(list)
            for r in selected:
                pred = int(np.argmax(r['scores']))
                confusion[r['target'], pred] += 1
                grouped[r['group_id']].append(pred == r['target'])
            expected = {'n': len(selected), 'confusion': confusion.tolist(),
                        'prediction_counts': confusion.sum(0).tolist(),
                        'accuracy_percent': 100 * float(np.trace(confusion)/len(selected)),
                        'all_group_correct_percent': 100 * float(np.mean([all(v) for v in grouped.values()]))}
            if task == 'counting':
                expected['mean_signed_count_error'] = float(np.mean([int(np.argmax(r['scores']))-r['target'] for r in selected]))
            equal(expected, stats, task)
        with np.load(cat / 'scores.npz', allow_pickle=False) as z:
            rows, shifts, by_class = score_diagnostics(dict(z), [r['target_id'] for r in read_json(cat/'samples.json')], read_json(cat/'classes.json'))
        equal(shifts, stored['category_score_shifts'])
        for filename, expected in [('category_sample_scores.csv', rows), ('category_class_scores.csv', by_class), ('category_transition_scores.csv', shifts)]:
            if [{k: str(v) for k, v in r.items()} for r in expected] != csv_rows(out / filename):
                raise ValueError(f'Historical score diagnostic mismatch: {filename}')
        result['metrics_recomputed'] = True
    else:
        raise ValueError('Unknown stage.')
    path = out.parent / (out.name + '_audit.json')
    write_json(path, result)
    print(json.dumps(result), flush=True)
    return result


def audit(run=None):
    paths = [ROOT / run] if run else sorted((ROOT / 'results').glob('*_depth_*_v1'))
    for out in paths:
        if not out.is_dir() or not out.resolve().is_relative_to(ROOT):
            raise ValueError('Invalid run directory.')
        audit_one(out)
