"""Freeze user-confirmed reviews and replay COCO sensitivity without inference."""
import argparse
import csv
import shutil
import hashlib
import zipfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from .depth.common import ROOT, read_json, write_csv, cluster_ci
from .depth.confirmation import paired_metrics
from .extension_artifacts import verify_manifest, close_enough
from .human_review import validate
from .io_utils import sha256, write_json, source_snapshot
from .spatial_diagnostics.natural_regions import summarize, audit as region_audit

REVIEW = ROOT / 'reviews/human_review_final_v1'
OUTPUT = ROOT / 'results/human_review_followup_v1'
REPORT = ROOT / 'reports/human_review_followup_v1'
PROTOCOL = ROOT / 'docs/human_review_followup_v1.md'
REGION = ROOT / 'results/full_natural_regions_confirmation_review_v2'
HOLISTIC = ROOT / 'results/full_natural_confirmation_v1'
VI = ROOT / 'results/full_vietnamese_cifar100_v1'


def read_sheet(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def normalize_sheet(source, destination, kind):
    """Remove formatting only; reject populated unknown columns and semantic changes."""
    template = ROOT / 'reviews/human_review_v2' / (
        'confirmation_100_blank.csv' if kind == 'images' else 'vietnamese_100_blank.csv')
    with template.open(encoding='utf-8-sig', newline='') as f:
        fields = csv.DictReader(f).fieldnames
    rows = read_sheet(source)
    normalized = []
    for row in rows:
        if any(k not in fields and (v if isinstance(v, list) else [v]) and
               any(str(x or '').strip() for x in (v if isinstance(v, list) else [v]))
               for k, v in row.items()):
            raise ValueError('Populated unexpected column')
        if not set(fields).issubset(row):
            raise ValueError('Missing source columns')
        normalized.append({k: (row[k] or '').strip() for k in fields})
    with destination.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(normalized)
    validate(destination, kind)
    return normalized


def freeze():
    if REVIEW.exists():
        raise FileExistsError('Frozen reviews already exist')
    REVIEW.mkdir()
    history = REVIEW / 'history'; history.mkdir()
    package = ROOT / 'data/human_review_v2'
    for source_name, name, kind in [
        ('confirmation_100_blank.csv', 'confirmation_100_R1.csv', 'images'),
        ('vietnamese_100_excel_utf8.csv', 'vietnamese_100_R1.csv', 'translations')]:
        source = package / source_name
        shutil.copyfile(source, history / ('received_' + source_name))
        rows = normalize_sheet(source, REVIEW / name, kind)
        counts = Counter(r['overall_accept' if kind == 'images' else 'status'] for r in rows)
        if counts != ({'yes': 98, 'no': 2} if kind == 'images' else {'accepted': 100}):
            raise ValueError('Unexpected final decisions: inspect before analysis')
        if any(r['annotator_id'] != 'R1' for r in rows):
            raise ValueError('Unexpected reviewer')
    for dirname in ['assessment_20261116', 'completion_check_20261116_192316']:
        if (package / dirname).exists():
            for p in sorted((package / dirname).iterdir()):
                if p.suffix in ['.csv', '.json', '.md']:
                    dest = history / dirname / p.name
                    dest.parent.mkdir(exist_ok=True); shutil.copyfile(p, dest)
    for p in package.glob('vietnamese_100_excel_utf8_before_acceptance_*.csv'):
        shutil.copyfile(p, history / p.name)
    shutil.copyfile(ROOT / 'reviews/human_review_v2/status.json', history / 'previous_status.json')
    shutil.copyfile(PROTOCOL, REVIEW / 'protocol.md')
    manifest = {'status': 'frozen', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                'reviewer': 'R1', 'basis': 'Explicit user confirmation; one human with agent discussion.',
                'assistant_actions': 'Formatting normalization and translation status entry explicitly authorized by user.',
                'sha256': {p.relative_to(REVIEW).as_posix(): sha256(p)
                           for p in sorted(REVIEW.rglob('*')) if p.is_file()}}
    write_json(REVIEW / 'freeze.json', manifest)
    print('Frozen reviews:', REVIEW)


def check_freeze():
    manifest = read_json(REVIEW / 'freeze.json')
    inventory = {p.relative_to(REVIEW).as_posix() for p in REVIEW.rglob('*')
                 if p.is_file() and p.name != 'freeze.json'}
    if inventory != set(manifest['sha256']): raise ValueError('Review inventory changed')
    for name, digest in manifest['sha256'].items():
        if sha256(REVIEW / name) != digest: raise ValueError('Review changed: ' + name)
    validate(REVIEW / 'confirmation_100_R1.csv', 'images')
    validate(REVIEW / 'vietnamese_100_R1.csv', 'translations')


def subset_indices(rows, keep):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['group_id'], []).append(i)
    if not keep or not keep.issubset(groups): raise ValueError('Empty or unknown sources')
    for group, indices in groups.items():
        if len(indices) != 2 or {rows[i]['variant'] for i in indices} != {'original', 'horizontal_flip'}:
            raise ValueError('Incomplete original/mirror source pair: ' + group)
        if {rows[i]['target'] for i in indices} != {0, 1}:
            raise ValueError('Incorrect source relations')
    return np.array([i for i, row in enumerate(rows) if row['group_id'] in keep])


def calculate():
    check_freeze()
    for run in [REGION, HOLISTIC, VI]: verify_manifest(run)
    region_audit(REGION)
    sheet = read_sheet(REVIEW / 'confirmation_100_R1.csv')
    key_file = OUTPUT / 'blind_source_mapping.json'
    if not key_file.exists():
        key_file = ROOT / 'data/human_review_v2/coordinator_private/answer_key_confirmation_100.json'
    key = read_json(key_file)
    # Only this deterministic mapping is used; labels/decisions are frozen separately.
    selected = read_json(ROOT / 'results/natural_confirmation_selection_v1/selected.json')
    lookup = {x['blind_id']: x for x in key['items']}
    if len(lookup) != 100 or set(lookup) != {r['blind_id'] for r in sheet}: raise ValueError('ID map')
    source_map = {r['source_image_id']: r for r in selected}
    membership = []
    for row in sheet:
        k = lookup[row['blind_id']]; s = source_map[k['source_image_id']]
        if k['image_sha256'] != s['image_sha256'] or row['object_a'] != s['category_a'] or row['object_b'] != s['category_b']:
            raise ValueError('Source mapping mismatch')
        membership.append({'blind_id': row['blind_id'], 'group_id': k['group_id'],
                           'source_image_id': k['source_image_id'], 'image_sha256': k['image_sha256'],
                           'accepted': row['overall_accept'] == 'yes', 'exclusion_reason': row['exclusion_reason']})
    keep = {m['group_id'] for m in membership if m['accepted']}
    if len(keep) != 98: raise ValueError('Expected 98 retained sources')
    rows = read_json(REGION / 'samples.json'); cfg = read_json(REGION / 'study_config.json')
    with np.load(REGION / 'scores.npz', allow_pickle=False) as z:
        scores = {k: z[k] for k in ['relational', 'padded_relational', 'holistic']}
    trials = read_json(HOLISTIC / 'trials.json'); old_cfg = read_json(HOLISTIC / 'config.json')
    with np.load(HOLISTIC / 'scores.npz', allow_pickle=False) as z: old_scores = z['scores']
    all_metrics, all_records = paired_metrics(trials, old_scores, old_cfg)
    close_enough(all_metrics, read_json(HOLISTIC / 'metrics.json'))
    filtered_metrics, filtered_records = paired_metrics(trials, old_scores, old_cfg, keep_ids=keep)
    bundles = {}; traces = {}
    for label, groups in [('before_100', {m['group_id'] for m in membership}), ('after_98', keep)]:
        ix = subset_indices(rows, groups); subset = [rows[i] for i in ix]
        bundles[label] = {}
        for method, field in [('holistic', 'holistic'), ('native_half', 'relational'), ('padded_half', 'padded_relational')]:
            result, predictions, pairs = summarize(subset, scores[field][ix], scores['holistic'][ix], 2, cfg)
            correct = [p['correct'] for p in predictions]
            result['both_images_correct_ci95_percent'] = cluster_ci(
                [p['both_images_correct'] for p in pairs], [p['group_id'] for p in pairs], cfg['bootstrap_draws'], cfg['bootstrap_seed'])
            result['correct_images'] = sum(correct)
            result['assignment_correct_sources'] = sum(p['correct'] for p in pairs)
            bundles[label][method] = result
            traces[f'{label}_{method}_predictions.csv'] = predictions
            traces[f'{label}_{method}_pairs.csv'] = pairs
        if label == 'before_100':
            old = read_json(REGION / 'metrics.json')
            for method, expected in [('native_half', {k:v for k,v in old.items() if k != 'secondary_padded_half'}), ('padded_half', old['secondary_padded_half'])]:
                close_enough({k:bundles[label][method][k] for k in expected}, expected)
    for label, m in [('before_100', all_metrics), ('after_98', filtered_metrics)]:
        close_enough(bundles[label]['holistic']['primary_accuracy_percent'], m['t2_o0']['two_way_accuracy_percent'])
        close_enough(bundles[label]['holistic']['horizontal_assignment']['accuracy_percent'], m['t2_o0']['accuracy_percent'])
    translations = read_sheet(REVIEW / 'vietnamese_100_R1.csv')
    lexicon = {str(r['class_id']):r for r in read_json(VI / 'multilingual_classes.json')}
    if any(r['vi_label_current'] != lexicon[r['class_id']]['vi_label'] or r['en_label'] != lexicon[r['class_id']]['en_label'] or r['status'] != 'accepted' for r in translations):
        raise ValueError('Translation lexicon changed')
    vi_link = {'human_review': '100 accepted by R1; user confirmation, assistant entered authorized statuses.',
               'review_sha256': sha256(REVIEW / 'vietnamese_100_R1.csv'),
               'lexicon_path': str((VI / 'multilingual_classes.json').relative_to(ROOT)),
               'lexicon_sha256': sha256(VI / 'multilingual_classes.json'),
               'metrics_path': str((VI / 'metrics.json').relative_to(ROOT)), 'metrics_sha256': sha256(VI / 'metrics.json'),
               'translations_unchanged': True, 'new_inference': False,
               'accuracy_percent': {k:v['accuracy_percent'] for k,v in read_json(VI / 'metrics.json')['single_templates'].items()}}
    return {'region_comparison': bundles, 'holistic_all_conditions_before': all_metrics,
            'holistic_all_conditions_after': filtered_metrics, 'translation_review': vi_link}, membership, traces, all_records, filtered_records


def run():
    if OUTPUT.exists() or REPORT.exists(): raise FileExistsError('Follow-up exists; use audit')
    metrics, membership, traces, before_pairs, after_pairs = calculate()
    OUTPUT.mkdir(); REPORT.mkdir()
    write_json(OUTPUT / 'metrics.json', metrics); write_csv(OUTPUT / 'source_membership.csv', membership)
    for name, records in traces.items(): write_csv(OUTPUT / name, records)
    write_csv(OUTPUT / 'holistic_all_conditions_before_pairs.csv', before_pairs)
    write_csv(OUTPUT / 'holistic_all_conditions_after_pairs.csv', after_pairs)
    comparison = []
    for method in ['holistic', 'native_half', 'padded_half']:
        for label in ['before_100', 'after_98']:
            m = metrics['region_comparison'][label][method]
            comparison.append({'subset': label, 'method': method, 'n_sources': m['n_sources'], 'n_images': m['n_images'],
                               'correct_images':m['correct_images'], 'top1_percent':m['primary_accuracy_percent'],
                               'top1_ci_low':m['primary_ci95_percent'][0], 'top1_ci_high':m['primary_ci95_percent'][1],
                               'both_correct_percent':m['both_images_correct_percent'],
                               'assignment_percent':m['horizontal_assignment']['accuracy_percent'],
                               'assignment_ci_low':m['horizontal_assignment']['ci95_percent'][0],
                               'assignment_ci_high':m['horizontal_assignment']['ci95_percent'][1],
                               'delta_vs_holistic_pp':m['paired_delta_pp'],
                               'delta_ci_low':m['paired_delta_ci95_pp'][0], 'delta_ci_high':m['paired_delta_ci95_pp'][1]})
    write_csv(REPORT / 'comparison.csv', comparison)
    write_json(REPORT / 'translation_review.json', metrics['translation_review'])
    # Manifest includes every frozen review, protocol, mapping and source artifact used.
    inputs = [PROTOCOL, REVIEW / 'freeze.json', *(p for p in REVIEW.rglob('*') if p.is_file()),
              ROOT / 'data/human_review_v2/coordinator_private/answer_key_confirmation_100.json',
              ROOT / 'results/natural_confirmation_selection_v1/selected.json']
    # Save a portable mapping without source pixels or private absolute paths.
    key_path = OUTPUT / 'blind_source_mapping.json'
    write_json(key_path, read_json(inputs[-2]))
    inputs.remove(inputs[-2]); inputs.append(key_path)
    for folder in [REGION, HOLISTIC, VI]:
        inputs.extend(p for p in folder.iterdir() if p.is_file())
    amendment = ROOT / 'docs/human_review_followup_v1_amendment.md'
    inputs.append(amendment)
    hashes, snapshot = source_snapshot(ROOT, [PROTOCOL, amendment, *(p for p in REVIEW.rglob('*') if p.is_file())])
    manifest = {'status':'completed', 'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'new_inference':False, 'posthoc_sensitivity':True, 'human_reviewers':1,
                'input_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in set(inputs)},
                'source_sha256':hashes, 'source_snapshot':snapshot}
    write_json(OUTPUT / 'manifest.json', manifest)
    print('Calculated before/after results:', OUTPUT)


def audit():
    manifest = read_json(OUTPUT / 'manifest.json')
    snap = manifest['source_snapshot']
    if sha256(ROOT / snap['path']) != snap['sha256']: raise ValueError('Source ZIP changed')
    with zipfile.ZipFile(ROOT / snap['path']) as archive:
        if set(archive.namelist()) != set(manifest['source_sha256']): raise ValueError('Source inventory')
        for name, digest in manifest['source_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest: raise ValueError('Archived source changed')
        for name, digest in manifest['input_sha256'].items():
            actual = hashlib.sha256(archive.read(name)).hexdigest() if name in archive.namelist() else sha256(ROOT / name)
            if actual != digest: raise ValueError('Changed input: ' + name)
    metrics, membership, traces, before, after = calculate()
    close_enough(metrics, read_json(OUTPUT / 'metrics.json'))
    for name, records in [('source_membership.csv', membership), *traces.items(),
                          ('holistic_all_conditions_before_pairs.csv', before), ('holistic_all_conditions_after_pairs.csv', after)]:
        if read_sheet(OUTPUT / name) != [{k:str(v) for k,v in r.items()} for r in records]:
            raise ValueError('Changed trace: ' + name)
    if 'artifact_sha256' in manifest:
        for name, digest in manifest['artifact_sha256'].items():
            if sha256(ROOT / name) != digest: raise ValueError('Changed artifact: ' + name)
    if (REPORT / 'sources.json').exists():
        sources=read_json(REPORT / 'sources.json')
        for name,digest in sources['input_sha256'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Report input changed')
        for name,digest in sources['derived_sha256'].items():
            if sha256(REPORT/name)!=digest:raise ValueError('Report changed')
    print('Audit passed: frozen reviews, all source hashes, metrics and per-image/source traces.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze','run','audit'])
    {'freeze':freeze, 'run':run, 'audit':audit}[parser.parse_args().action]()
