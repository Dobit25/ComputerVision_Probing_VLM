"""Run a frozen paired experiment using shared image features and 100 candidates."""

import csv
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone

from ..configuration import validate_config
from ..dataset_adapters import load_dataset
from ..io_utils import configure_paths, inside_repo, sha256, source_snapshot, write_json, command_output
from ..model_adapters import load_model
from ..verify import require, verify as verify_baseline
from .metrics import summarize
from .taxonomy import STUDY, load_classes


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def get_predictions(scores, count):
    import numpy as np
    result = {}
    require(set(scores) == {'baseline', 'category'}, 'Both score conditions are required.')
    for name, matrix in scores.items():
        require(matrix.shape == (count, 100) and matrix.dtype == np.float32, 'Expected N x 100 FP32 scores.')
        require(np.isfinite(matrix).all() and (abs(matrix) <= 1.00001).all(), 'Invalid cosine scores.')
        result[name] = matrix.argmax(axis=1).tolist()
    return result


def prediction_rows(samples, predictions):
    records = []
    for i, sample in enumerate(samples):
        b, c, y = predictions['baseline'][i], predictions['category'][i], sample['target_id']
        transition = ('both_correct' if b == y else 'wrong_to_correct') if c == y else (
            'correct_to_wrong' if b == y else 'both_wrong')
        records.append({**sample, 'baseline_prediction_id': b, 'category_prediction_id': c,
                        'baseline_correct': int(b == y), 'category_correct': int(c == y), 'transition': transition})
    return records


def check_parity(root, samples, scores, predictions):
    import numpy as np
    baseline = root / STUDY['baseline_run']
    verify_baseline(baseline)
    refs = csv_rows(baseline / 'predictions.csv')
    maximum = 0.0
    for i, sample in enumerate(samples):
        ref, matrix_row = refs[i], scores['baseline'][i]
        require(sample == {'test_index': i, 'sample_id': ref['sample_id'], 'target_id': int(ref['target_id'])},
                'Input differs from historical baseline.')
        pred = predictions['baseline'][i]
        require(pred == int(ref['prediction_id']), 'Baseline prediction changed.')
        top_two = np.sort(matrix_row)[-2:]
        measured = [float(matrix_row[pred]), float(matrix_row[sample['target_id']]), float(top_two[1] - top_two[0])]
        recorded = [float(ref[k]) for k in ('predicted_cosine', 'target_cosine', 'top1_top2_margin')]
        error = max(abs(a - b) for a, b in zip(measured, recorded))
        require(error <= 1e-6, 'Baseline cosine/margin changed beyond tolerance.')
        maximum = max(maximum, error)
    return {'predictions_match': True, 'compared_samples': len(samples),
            'max_absolute_cosine_or_margin_difference': maximum, 'tolerance': 1e-6}


def run(root, run_name, smoke=False):
    config_path = root / 'configs/cifar100_category_v1.json'
    study = json.loads(config_path.read_text())
    require(study == STUDY, 'Study differs from frozen v1 protocol.')
    model = json.loads((root / study['base_config']).read_text())
    validate_config(model)
    reference = json.loads((root / study['baseline_run'] / 'config.json').read_text())
    require(model == {k: v for k, v in reference.items() if k not in ('limit', 'selection')},
            'Model/data settings differ from baseline.')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', run_name), 'Invalid run name.')
    output = root / 'results' / run_name
    require(not output.exists(), 'Run directory already exists; choose a new name.')
    configure_paths(root)
    output.mkdir(parents=True)
    config = {'study': study, 'model': model, 'limit': 200 if smoke else None}
    source_hashes, snapshot = source_snapshot(root, [config_path, root / study['base_config'],
                                                   root / 'docs/cifar100_category_protocol.md'])
    manifest = {'status': 'running', 'kind': 'cifar100_category_v1', 'mode': 'smoke' if smoke else 'full',
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'command_argv': sys.argv,
                'git_head': command_output(['git', 'rev-parse', 'HEAD'], root),
                'source_sha256': source_hashes, 'source_snapshot': snapshot,
                'baseline_manifest_sha256': sha256(root / study['baseline_run'] / 'manifest.json')}
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', manifest)
    start = time.perf_counter()
    try:
        evaluate(root, output, config, manifest)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - start)
        write_json(output / 'manifest.json', manifest)


def evaluate(root, output, config, manifest):
    print(f'Category study: {output}; loading libraries...', flush=True)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    cfg = config['model']
    require(torch.cuda.is_available(), 'CUDA required by the frozen baseline protocol.')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    write_json(output / 'environment.json', {'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'packages': dict(sorted(packages.items())), 'device': 'cuda',
        'gpu': torch.cuda.get_device_name(0), 'torch_cuda_build': torch.version.cuda,
        'gpu_driver': command_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], root),
        'precision': 'fp32', 'tf32': False, 'deterministic_algorithms': True, 'cpu_threads': torch.get_num_threads(),
        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']})
    adapter = load_model(root, cfg, torch.device('cuda'))
    bundle = load_dataset(root, cfg, adapter.preprocess)
    classes, provenance = load_classes(root, config['study'])
    require(bundle.classes == [row['class_name'] for row in classes], 'Taxonomy and dataset class order differ.')
    count = config['limit'] or len(bundle.images)
    samples = [{'test_index': i, 'sample_id': bundle.sample_ids[i], 'target_id': bundle.targets[i]} for i in range(count)]
    write_json(output / 'classes.json', classes)
    write_json(output / 'samples.json', samples)
    manifest.update(adapter.metadata)
    manifest.update(dataset=bundle.metadata, taxonomy=provenance, num_samples=count, candidate_count=100)
    write_json(output / 'manifest.json', manifest)

    def normalized(features):
        require(features.ndim == 2 and features.shape[1] == adapter.embedding_dim, 'Embedding shape mismatch.')
        require(torch.isfinite(features).all() and (features.norm(dim=-1) > 0).all(), 'Invalid embeddings.')
        return torch.nn.functional.normalize(features.float(), dim=-1)

    scores = {name: np.empty((count, 100), dtype=np.float32) for name in ('baseline', 'category')}
    loader = DataLoader(Subset(bundle.images, range(count)), batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=0, pin_memory=True, drop_last=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        text = {name: normalized(adapter.encode_texts([r[name + '_prompt'] for r in classes])) for name in scores}
        offset = 0
        for images, labels in loader:
            require(labels.tolist() == bundle.targets[offset:offset + len(labels)], 'Input label order changed.')
            features = normalized(adapter.encode_images(images.to('cuda', non_blocking=True)))
            for name in scores:
                scores[name][offset:offset + len(labels)] = (features @ text[name].T).cpu().numpy()
            offset += len(labels)
            if offset % 640 == 0 or offset == count:
                print(f'Both conditions: {offset}/{count}', flush=True)
    torch.cuda.synchronize()
    manifest.update(evaluation_seconds=time.perf_counter() - start,
                    peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated())
    predictions = get_predictions(scores, count)
    manifest['baseline_parity'] = check_parity(root, samples, scores, predictions)
    np.savez_compressed(output / 'scores.npz', **scores)
    write_csv(output / 'predictions.csv', prediction_rows(samples, predictions))
    metrics, per_class, groups, confusions = summarize(bundle.targets[:count], predictions['baseline'],
        predictions['category'], classes, config['study']['bootstrap_draws'], config['study']['bootstrap_seed'])
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'confusion_matrices.json', confusions)
    write_csv(output / 'per_class.csv', per_class)
    write_csv(output / 'per_category.csv', groups)
    manifest['artifact_sha256'] = {p.name: sha256(p) for p in output.iterdir() if p.name != 'manifest.json'}
    print({name: metrics[name]['top1_accuracy_percent'] for name in ('baseline', 'category')}, flush=True)

