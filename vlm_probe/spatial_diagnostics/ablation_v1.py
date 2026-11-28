"""T2 (1124.md, WS1 part 2): spatial ablation and oracle bbox baseline on COCO Sets A and B.

    run    GPU: holistic full-image padded T2/o0 scores, after a center-crop parity gate
    audit  offline: recompute all conditions from saved scores and frozen artifacts

Conditions (see docs/spatial_ablation_protocol_v1.md):
    1 holistic_center_crop   frozen T2/o0 scores (reuse)
    2 holistic_full_padded   new inference: whole image on a square canvas, no crop
    3 native_half / padded_half   reuse the regions runs
    4 bbox_centroid / bbox_ideal_half   oracle COCO boxes on the 224 baseline canvas
    5 swapped_region         native region scores with left<->right and top<->bottom swapped
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..depth.common import ROOT, csv_rows, finish, read_json, write_csv
from ..io_utils import sha256, write_json
from .natural_regions import holistic_scores, summarize
from .quadrant_mitigation import _box_half_overlap, relational_scores_from_regions

CONFIG = ROOT / 'configs/spatial_ablation_v1.json'
SETS = ('B', 'A')
SWAP = [1, 0, 3, 2]  # (left, right, top, bottom) -> (right, left, bottom, top)
CONDITIONS = ('holistic_center_crop', 'holistic_full_padded', 'native_half', 'padded_half',
              'bbox_centroid', 'bbox_ideal_half', 'swapped_region')


# ---------------------------------------------------------------- inputs

def set_rows(cfg, key, smoke=False):
    from ..depth import natural
    selection = read_json(ROOT / cfg['sets'][key]['selection'])
    if smoke:
        selection = selection[:cfg['smoke_sources_per_set']]
    return selection, natural.make_images(selection)


def cohort_mask(cfg, key, rows):
    """Set B keeps the reviewed sources (accepted == True); Set A keeps every source."""
    path = cfg['sets'][key]['membership']
    if not path:
        return np.ones(len(rows), dtype=bool)
    accepted = {r['group_id'] for r in csv_rows(ROOT / path) if r['accepted'] == 'True'}
    return np.array([r['group_id'] in accepted for r in rows], dtype=bool)


def frozen_center(cfg, key, rows):
    return holistic_scores(rows, {'baseline': cfg['sets'][key]['baseline']}).astype(np.float32)


def region_scores(cfg, key, rows):
    run = ROOT / cfg['sets'][key]['regions_run']
    index = {s['image_id']: i for i, s in enumerate(read_json(run / 'samples.json'))}
    take = [index[r['image_id']] for r in rows]
    with np.load(run / 'scores.npz', allow_pickle=False) as z:
        return {k: z[k][take] for k in ('regions', 'relational', 'padded_regions', 'padded_relational')}


def holistic_trials(cfg, key, rows):
    """T2/o0 trials of the baseline run, one per row and in row order."""
    ref = ROOT / cfg['sets'][key]['baseline']
    ref_index = {r['image_id']: i for i, r in enumerate(read_json(ref / 'images.json'))}
    by_image = {t['image_index']: t for t in read_json(ref / 'trials.json')
                if t['template'] == cfg['holistic_template'] and t['order'] == cfg['holistic_order']}
    return [by_image[ref_index[r['image_id']]] for r in rows]


# ---------------------------------------------------------------- conditions

def pad_full(image, rgb):
    """Whole image centered on a square canvas of side max(w, h); nothing is cropped."""
    from PIL import Image
    side = max(image.size)
    canvas = Image.new('RGB', (side, side), tuple(rgb))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def canvas_boxes(row, size):
    """Boxes of A and B on the baseline 224 canvas, flipping raw coordinates first for mirrored images."""
    from .natural_recognition import canvas_box
    boxes = []
    for key in ('bbox_a', 'bbox_b'):
        x, y, w, h = [float(v) for v in row[key]]
        if row['transform'] == 'horizontal_flip':
            x = row['width'] - x - w
        boxes.append(canvas_box((x, y, w, h), row['width'], row['height'], size))
    return boxes


def bbox_scores(rows, size, threshold):
    n = len(rows)
    centroid, half = np.zeros((n, 4)), np.zeros((n, 4))
    same_half = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows):
        a, b = canvas_boxes(r, size)
        dx = 0.5 * (a[0] + a[2]) - 0.5 * (b[0] + b[2])
        dy = 0.5 * (a[1] + a[3]) - 0.5 * (b[1] + b[3])
        centroid[i] = [-dx, dx, -dy, dy]
        sa = _box_half_overlap([a[0], a[1], max(1e-6, a[2] - a[0]), max(1e-6, a[3] - a[1])], size, size)
        sb = _box_half_overlap([b[0], b[1], max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])], size, size)
        half[i] = relational_scores_from_regions(sa, sb)
        axis = (0, 1) if int(r['target']) in (0, 1) else (2, 3)
        same_half[i] = any(sa[k] > threshold and sb[k] > threshold for k in axis)
    return centroid, half, same_half


def swapped_scores(regions):
    return relational_scores_from_regions(regions[:, SWAP, 0], regions[:, SWAP, 1])


# ---------------------------------------------------------------- metrics

def _pct(flags):
    flags = np.asarray(flags, dtype=bool)
    return float(100 * flags.mean()) if flags.size else None


def set_metrics(cfg, key, rows, padded):
    entry = cfg['sets'][key]
    choices = entry['primary_choices']
    mask = cohort_mask(cfg, key, rows)
    idx = np.flatnonzero(mask)
    sub = [rows[i] for i in idx]
    center = frozen_center(cfg, key, rows)
    reg = region_scores(cfg, key, rows)
    if not np.allclose(relational_scores_from_regions(reg['regions'][:, :, 0], reg['regions'][:, :, 1]),
                       reg['relational'], atol=1e-6):
        raise ValueError(f'Set {key}: stored native relational scores do not match their region scores.')
    centroid, half, same = bbox_scores(rows, cfg['canvas_size'], cfg['same_half_threshold'])
    scores = {'holistic_center_crop': center, 'holistic_full_padded': padded,
              'native_half': reg['relational'], 'padded_half': reg['padded_relational'],
              'bbox_centroid': centroid, 'bbox_ideal_half': half, 'swapped_region': swapped_scores(reg['regions'])}
    conditions, predictions = {}, {}
    for name in CONDITIONS:
        result, preds, _ = summarize(sub, np.asarray(scores[name], dtype=np.float64)[idx], center[idx], choices, cfg)
        result.pop('primary_confusion', None)
        conditions[name] = result
        predictions[name] = [p['prediction'] for p in preds]
    y = np.array([r['target'] for r in sub])
    same_sub = same[idx]
    native_ok = np.asarray(scores['native_half'])[idx][:, :choices].argmax(1) == y
    ideal_ok = half[idx][:, :choices].argmax(1) == y
    d_pad = conditions['holistic_full_padded']['paired_delta_pp']
    d_native = conditions['native_half']['paired_delta_pp']
    return {
        'set': key, 'name': entry['name'], 'role': entry['role'], 'primary_choices': choices,
        'n_sources': len({r['group_id'] for r in sub}), 'n_images': len(sub),
        'conditions': conditions,
        'same_half': {'threshold': cfg['same_half_threshold'], 'n_images': int(same_sub.sum()),
                      'percent_of_images': _pct(same_sub),
                      'native_accuracy_same_half_percent': _pct(native_ok[same_sub]),
                      'native_accuracy_other_percent': _pct(native_ok[~same_sub]),
                      'ideal_half_accuracy_same_half_percent': _pct(ideal_ok[same_sub]),
                      'ideal_half_accuracy_other_percent': _pct(ideal_ok[~same_sub])},
        'gap_closed_by_padding_fraction': float(d_pad / d_native) if d_native else None,
    }, {'image_id': [r['image_id'] for r in sub], 'target': y.tolist(), **predictions}


def interpret(cfg, primary):
    lo, hi = primary['paired_delta_ci95_pp']
    rule = cfg['interpretation']
    if hi < rule['center_crop_not_main_cause_if_ci_upper_below_pp']:
        return 'center_crop_not_main_cause'
    if lo >= rule['center_crop_contributes_if_ci_lower_at_least_pp']:
        return 'center_crop_contributes'
    return 'unclear'


def build_metrics(cfg, data, padded, parity):
    sets, prediction_rows = {}, []
    for key in SETS:
        sets[key], preds = set_metrics(cfg, key, data[key], padded[key])
        for i, image_id in enumerate(preds['image_id']):
            prediction_rows.append({'set': key, 'image_id': image_id, 'target': preds['target'][i],
                                    **{c: preds[c][i] for c in CONDITIONS}})
    primary = sets['B']['conditions']['holistic_full_padded']
    metrics = {'study': cfg['study'], 'sets': sets, 'parity': parity,
               'primary_comparison': {**cfg['primary_comparison'],
                                      'delta_pp': primary['paired_delta_pp'],
                                      'ci95_pp': primary['paired_delta_ci95_pp'],
                                      'verdict': interpret(cfg, primary)}}
    return metrics, prediction_rows


# ---------------------------------------------------------------- provenance

def protocol_commit(cfg):
    files = [cfg['protocol'], CONFIG.relative_to(ROOT).as_posix()]
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if git('status', '--porcelain', '--', *files):
        raise RuntimeError('Protocol/config have uncommitted changes; commit them before inference.')
    log = git('log', '-1', '--format=%H %cI', '--', *files).split()
    if len(log) != 2:
        raise RuntimeError('Protocol/config are not committed; inference refused.')
    when = datetime.fromisoformat(log[1]).astimezone(timezone.utc)
    return {'commit': log[0], 'commit_time_utc': when.isoformat(), 'files_sha256': {f: sha256(ROOT / f) for f in files}}


def _inputs(cfg):
    paths = [ROOT / cfg['protocol'], ROOT / cfg['regions_config']]
    for key in SETS:
        e = cfg['sets'][key]
        paths += [ROOT / e['selection'], ROOT / e['baseline'] / 'manifest.json', ROOT / e['baseline'] / 'scores.npz',
                  ROOT / e['baseline'] / 'trials.json', ROOT / e['baseline'] / 'images.json',
                  ROOT / e['regions_run'] / 'manifest.json', ROOT / e['regions_run'] / 'scores.npz',
                  ROOT / e['regions_run'] / 'samples.json']
        if e['membership']:
            paths.append(ROOT / e['membership'])
    return paths


# ---------------------------------------------------------------- run / audit

def run(cfg, smoke=False, run_name=None):
    if cfg['status'] != 'frozen':
        raise RuntimeError('Config is not frozen; inference refused (protocol section 7).')
    provenance = protocol_commit(cfg)
    from ..depth import natural
    from ..depth.common import Engine, ImageFiles
    from ..extension_artifacts import begin_extension
    data = {}
    for key in SETS:
        selection, data[key] = set_rows(cfg, key, smoke)
        for s in selection:
            natural.download_image(s)
    name = run_name or f'{"smoke" if smoke else "full"}_spatial_ablation_v1'
    out, _, manifest = begin_extension('spatial_ablation', CONFIG, inputs=_inputs(cfg), smoke=smoke, run_name=name)
    if datetime.fromisoformat(provenance['commit_time_utc']) >= datetime.fromisoformat(manifest['started_at_utc']):
        raise ValueError('Protocol must be committed before inference starts.')
    manifest['protocol_commit'] = provenance
    engine = Engine(out, read_json(ROOT / cfg['base_config']), manifest)

    class PaddedFull:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            from PIL import Image, ImageOps
            row = self.rows[i]
            path = ROOT / row['image_path']
            if sha256(path) != row['image_sha256']:
                raise ValueError(f'Changed source image: {path}')
            with Image.open(path) as im:
                image = im.convert('RGB')
            if row['transform'] == 'horizontal_flip':
                image = ImageOps.mirror(image)
            return engine.adapter.preprocess(pad_full(image, cfg['pad_rgb']))

    saved, parity, padded = {}, {}, {}
    for key in SETS:
        rows = data[key]
        trials = holistic_trials(cfg, key, rows)
        pool = list(dict.fromkeys(c for t in trials for c in t['captions']))
        pix = {c: j for j, c in enumerate(pool)}
        text = engine.texts(pool)
        gather = lambda s: np.array([[s[i, pix[c]] for c in t['captions']] for i, t in enumerate(trials)], dtype=np.float32)
        identity = {'images': [{k: r.get(k) for k in ('image_id', 'image_sha256', 'transform')} for r in rows]}
        rebuilt = gather(engine.scores(engine.images(ImageFiles(rows, engine.adapter.preprocess), identity), text))
        diff = float(np.abs(rebuilt.astype(np.float64) - frozen_center(cfg, key, rows)).max())
        parity[key] = {'images': len(rows), 'max_abs_diff': diff, 'tolerance': cfg['parity_tolerance']}
        if diff > cfg['parity_tolerance']:
            raise ValueError(f'Set {key}: holistic center-crop parity failed: {parity[key]}')
        view = {**identity, 'view': 'full_image_padded_v1', 'pad_rgb': cfg['pad_rgb']}
        padded[key] = gather(engine.scores(engine.images(PaddedFull(rows), view), text))
        saved[f'{key}_center_rebuilt'] = rebuilt
        saved[f'{key}_holistic_full_padded'] = padded[key]
    metrics, predictions = build_metrics(cfg, data, padded, parity)
    write_json(out / 'images.json', data)
    np.savez_compressed(out / 'scores.npz', **saved)
    write_json(out / 'parity.json', parity)
    write_json(out / 'metrics.json', metrics)
    write_csv(out / 'predictions.csv', predictions)
    manifest.update(num_images={k: len(v) for k, v in data.items()})
    finish(out, manifest)
    print(json.dumps(metrics['primary_comparison'], indent=1), flush=True)


def audit(run_name=None, directory=None):
    from ..extension_artifacts import close_enough
    out = Path(directory) if directory else ROOT / 'results' / (run_name or 'full_spatial_ablation_v1')
    m = read_json(out / 'manifest.json')
    if m['status'] != 'completed':
        raise ValueError('T2 run incomplete.')
    if {p.name for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'} != set(m['artifact_sha256']):
        raise ValueError('Unexpected or missing T2 artifacts.')
    for name, digest in m['artifact_sha256'].items():
        if sha256(out / name) != digest:
            raise ValueError(f'Changed T2 artifact: {name}')
    for name, digest in m['input_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Changed T2 input: {name}')
    cfg = read_json(out / 'study_config.json')
    prov = m['protocol_commit']
    if datetime.fromisoformat(prov['commit_time_utc']) >= datetime.fromisoformat(m['started_at_utc']):
        raise ValueError('Protocol was not committed before inference.')
    for name, digest in prov['files_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Protocol/config changed after the run: {name}')
    smoke = m['mode'] == 'smoke'
    data = read_json(out / 'images.json')
    for key in SETS:
        if set_rows(cfg, key, smoke)[1] != data[key]:
            raise ValueError(f'Set {key}: images differ from the frozen selection.')
    with np.load(out / 'scores.npz', allow_pickle=False) as z:
        saved = {k: z[k] for k in z.files}
    parity, padded = {}, {}
    for key in SETS:
        diff = float(np.abs(saved[f'{key}_center_rebuilt'].astype(np.float64) - frozen_center(cfg, key, data[key])).max())
        if diff > cfg['parity_tolerance']:
            raise ValueError(f'Set {key}: saved parity exceeds tolerance.')
        parity[key] = {'images': len(data[key]), 'max_abs_diff': diff, 'tolerance': cfg['parity_tolerance']}
        padded[key] = saved[f'{key}_holistic_full_padded']
    close_enough(parity, read_json(out / 'parity.json'))
    metrics, predictions = build_metrics(cfg, data, padded, parity)
    close_enough(metrics, read_json(out / 'metrics.json'))
    if [{k: str(v) for k, v in r.items()} for r in predictions] != csv_rows(out / 'predictions.csv'):
        raise ValueError('T2 predictions differ from recomputation.')
    b = metrics['sets']['B']['conditions']
    result = {'verified': True, 'recomputed_from_scores': True, 'new_inference_in_audit': False,
              'protocol_committed_before_inference': True,
              'parity_max_abs_diff': {k: v['max_abs_diff'] for k, v in parity.items()},
              'primary_comparison': metrics['primary_comparison'],
              'set_b_top1': {c: b[c]['primary_accuracy_percent'] for c in CONDITIONS}}
    write_json(out.parent / (out.name + '_audit.json'), result)
    print(json.dumps(result, indent=1), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['run', 'audit'])
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--run-name')
    a = p.parse_args(argv)
    if a.action == 'run':
        run(read_json(CONFIG), a.smoke, a.run_name)
    else:
        audit(run_name=a.run_name)


if __name__ == '__main__':
    main()
