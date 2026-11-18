"""Recompute paired study results from all-class scores and verified CIFAR labels."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from ..configuration import validate_config
from ..io_utils import inside_repo, sha256
from ..verify import require
from .metrics import summarize
from .runner import csv_rows, get_predictions, prediction_rows, check_parity
from .taxonomy import STUDY, load_classes, read_source

ARTIFACTS = {'config.json', 'environment.json', 'classes.json', 'samples.json', 'scores.npz',
             'predictions.csv', 'metrics.json', 'per_class.csv', 'per_category.csv', 'confusion_matrices.json'}


def compare_csv(path, rows):
    expected = [{k: '' if value is None else str(value) for k, value in row.items()} for row in rows]
    require(csv_rows(path) == expected, f'CSV recomputation differs: {path.name}')


def verify(root, directory):
    import numpy as np

    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest['status'] == 'completed' and manifest['kind'] == 'cifar100_category_v1', 'Incomplete or wrong run kind.')
    require(set(manifest['artifact_sha256']) == ARTIFACTS, 'Missing/unexpected artifacts.')
    for name, digest in manifest['artifact_sha256'].items():
        require(sha256(directory / name) == digest, f'Artifact changed: {name}')
    cfg = json.loads((directory / 'config.json').read_text())
    require(cfg['study'] == STUDY, 'Study configuration changed.')
    validate_config(cfg['model'])
    ref_config = json.loads((root / STUDY['baseline_run'] / 'config.json').read_text())
    require(cfg['model'] == {k: v for k, v in ref_config.items() if k not in ('limit', 'selection')}, 'Baseline config mismatch.')
    require(manifest['mode'] in ('smoke', 'full'), 'Unknown run mode.')
    require(cfg['limit'] == (200 if manifest['mode'] == 'smoke' else None), 'Run mode and limit disagree.')
    require(manifest['checkpoint']['sha256'] == cfg['model']['checkpoint_sha256'] and
            manifest['checkpoint']['url'] == cfg['model']['checkpoint_url'] and
            manifest['checkpoint']['quick_gelu'] is True, 'Checkpoint mismatch.')
    require(manifest['model_embedding_dim'] == 512 and manifest['candidate_count'] == 100, 'Model/candidate dimensions mismatch.')
    baseline = root / STUDY['baseline_run']
    require(sha256(baseline / 'manifest.json') == manifest['baseline_manifest_sha256'], 'Historical reference changed.')
    snap = manifest['source_snapshot']
    path = inside_repo(root, snap['path'])
    require(sha256(path) == snap['sha256'], 'Source ZIP changed.')
    with zipfile.ZipFile(path) as archive:
        require(all(hashlib.sha256(archive.read(k)).hexdigest() == v for k, v in manifest['source_sha256'].items()),
                'Source ZIP content mismatch.')
    classes, provenance = load_classes(root, cfg['study'])
    require(classes == json.loads((directory / 'classes.json').read_text()), 'Taxonomy/prompts differ from train mapping.')
    require(provenance == manifest['taxonomy'], 'Taxonomy provenance differs.')
    test = read_source(root, 'test')
    count = cfg['limit'] or 10000
    samples = [{'test_index': i, 'sample_id': f'cifar100/test/{i:05d}', 'target_id': test['fine_labels'][i]} for i in range(count)]
    require(samples == json.loads((directory / 'samples.json').read_text()) and manifest['num_samples'] == count,
            'Test samples/labels differ from official source.')
    require(all(classes[fine]['coarse_id'] == coarse for fine, coarse in zip(test['fine_labels'], test['coarse_labels'])),
            'Train taxonomy does not agree with official test taxonomy.')
    with np.load(directory / 'scores.npz', allow_pickle=False) as archive:
        scores = {name: archive[name] for name in archive.files}
    predictions = get_predictions(scores, count)
    require(check_parity(root, samples, scores, predictions) == manifest['baseline_parity'], 'Parity record differs.')
    compare_csv(directory / 'predictions.csv', prediction_rows(samples, predictions))
    metrics, per_class, groups, confusions = summarize(test['fine_labels'][:count], predictions['baseline'],
        predictions['category'], classes, STUDY['bootstrap_draws'], STUDY['bootstrap_seed'])
    require(metrics == json.loads((directory / 'metrics.json').read_text()), 'Metrics/CI differ from scores.')
    require(confusions == json.loads((directory / 'confusion_matrices.json').read_text()), 'Confusion matrices differ.')
    compare_csv(directory / 'per_class.csv', per_class)
    compare_csv(directory / 'per_category.csv', groups)
    return {'verified': True, 'mode': manifest['mode'], 'num_samples': count, 'candidate_count': 100,
            'baseline_accuracy_percent': metrics['baseline']['top1_accuracy_percent'],
            'category_accuracy_percent': metrics['category']['top1_accuracy_percent'],
            'paired': metrics['paired'], 'baseline_parity': manifest['baseline_parity']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(verify(root, inside_repo(root, args.run_directory)), indent=2))


if __name__ == '__main__':
    main()
