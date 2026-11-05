"""Run provenance, deterministic inference, and paired uncertainty estimates."""
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..io_utils import configure_paths, sha256, source_snapshot, write_json, command_output

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/probing_depth_v1.json'
PROTOCOL = ROOT / 'docs/probing_depth_protocol.md'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty CSV is not an auditable result.')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def begin(stage, smoke=False, inputs=(), run_name=None):
    configure_paths(ROOT)
    import re
    name = run_name or f'{"smoke" if smoke else "full"}_depth_{stage}_v1'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', name):
        raise ValueError('Run name must be a simple directory name without path separators.')
    out = ROOT / 'results' / name
    if out.exists():
        raise FileExistsError(f'Frozen run exists: {out}')
    out.mkdir(parents=True)
    cfg = read_json(CONFIG)
    hashes, snapshot = source_snapshot(ROOT, [CONFIG, PROTOCOL, ROOT / cfg['model_config']])
    manifest = {'status': 'running', 'stage': stage, 'mode': 'smoke' if smoke else 'full',
                'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'git_head': command_output(['git', 'rev-parse', 'HEAD'], ROOT),
                'source_sha256': hashes, 'source_snapshot': snapshot,
                'input_sha256': {str(Path(p).relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in inputs}}
    write_json(out / 'config.json', cfg)
    (out / 'protocol.md').write_bytes(PROTOCOL.read_bytes())
    write_json(out / 'manifest.json', manifest)
    return out, cfg, manifest


def finish(out, manifest):
    manifest.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat())
    manifest['artifact_sha256'] = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.name != 'manifest.json' and p.is_file()}
    write_json(out / 'manifest.json', manifest)


def cluster_ci(values, groups, draws=2000, seed=1402):
    values = np.asarray(values, dtype=float)
    _, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=values)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(sizes), size=(draws, len(sizes)))
    sampled = totals[indices].sum(1) / sizes[indices].sum(1)
    return (100 * np.quantile(sampled, [.025, .975])).tolist()


def paired_class_ci(difference, labels, draws=2000, seed=1402):
    values = np.asarray(difference, dtype=float)
    labels = np.asarray(labels)
    strata = [values[labels == c] for c in np.unique(labels)]
    rng = np.random.default_rng(seed)
    totals = np.zeros(draws)
    for v in strata:
        indices = rng.integers(len(v), size=(draws, len(v)))
        totals += v[indices].sum(1)
    return (100 * np.quantile(totals / len(values), [.025, .975])).tolist()


def classification_stats(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=int)
    if scores.ndim != 2 or len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError('Invalid score matrix.')
    pred = scores.argmax(1)
    true = scores[np.arange(len(labels)), labels]
    other = scores.copy()
    other[np.arange(len(labels)), labels] = -np.inf
    rival = other.argmax(1)
    rival_score = other[np.arange(len(labels)), rival]
    return pred, pred == labels, true, rival, rival_score, true - rival_score


class Engine:
    def __init__(self, out, cfg, manifest):
        import importlib.metadata
        import random
        import torch
        from ..model_adapters import load_model
        self.torch = torch
        self.cfg = read_json(ROOT / cfg['model_config'])
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for baseline parity.')
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        self.adapter = load_model(ROOT, self.cfg, torch.device('cuda'))
        self.manifest = manifest
        manifest.update(self.adapter.metadata)
        write_json(out / 'environment.json', {
            'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'gpu': torch.cuda.get_device_name(0), 'torch_cuda': torch.version.cuda,
            'precision': 'fp32', 'batch_size': 32, 'deterministic_algorithms': True,
            'packages': {name: importlib.metadata.version(name) for name in
                         ['torch', 'torchvision', 'open_clip_torch', 'numpy', 'Pillow', 'matplotlib']}})

    def norm(self, features):
        t = self.torch
        if features.ndim != 2 or features.shape[1] != self.adapter.embedding_dim or not t.isfinite(features).all():
            raise ValueError('Invalid features.')
        if (features.norm(dim=-1) <= 0).any():
            raise ValueError('Zero feature.')
        return t.nn.functional.normalize(features.float(), dim=-1)

    def texts(self, texts, batch=32):
        with self.torch.inference_mode():
            return self.torch.cat([self.norm(self.adapter.encode_texts(texts[i:i+batch])).cpu()
                                   for i in range(0, len(texts), batch)])

    def images(self, dataset, identity):
        """Cache only exact model/preprocess/data/code identities; verify cache bytes."""
        import hashlib
        from torch.utils.data import DataLoader
        key_data = {'input': identity, 'checkpoint': self.manifest['checkpoint'],
                    'preprocessing': self.manifest['preprocessing'], 'precision': 'fp32',
                    'batch_size': 32, 'engine_source_sha256': sha256(Path(__file__))}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        folder = ROOT / '.cache/depth_features'
        folder.mkdir(exist_ok=True)
        path, meta = folder / f'{key}.npy', folder / f'{key}.json'
        if path.exists() and meta.exists():
            m = read_json(meta)
            if m['identity'] != key_data or sha256(path) != m['sha256']:
                raise ValueError('Feature cache provenance mismatch.')
            array = np.load(path, allow_pickle=False)
            print(f'Verified cached image features: {len(array)}', flush=True)
        else:
            features = []
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
            with self.torch.inference_mode():
                for batch in loader:
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    features.append(self.norm(self.adapter.encode_images(batch.to('cuda'))).cpu())
                    if len(features) % 20 == 0:
                        print(f'Image features: {min(32 * len(features), len(dataset))}/{len(dataset)}', flush=True)
            array = self.torch.cat(features).numpy()
            np.save(path, array, allow_pickle=False)
            write_json(meta, {'identity': key_data, 'sha256': sha256(path)})
        if array.shape != (len(dataset), self.adapter.embedding_dim) or not np.isfinite(array).all():
            raise ValueError('Feature cache shape mismatch.')
        if not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5):
            raise ValueError('Features are not normalized.')
        self.manifest.setdefault('feature_caches', []).append({'path': path.relative_to(ROOT).as_posix(),
                                                              'sha256': sha256(path), 'identity': key_data})
        return self.torch.from_numpy(array)

    def scores(self, images, texts):
        # Same 32-image GEMM geometry as baseline, including cached image features.
        with self.torch.inference_mode():
            text = texts.to('cuda')
            return np.concatenate([(images[i:i+32].to('cuda') @ text.T).cpu().numpy()
                                   for i in range(0, len(images), 32)])


class ImageFiles:
    def __init__(self, records, preprocess):
        self.records, self.preprocess = records, preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        from PIL import Image, ImageOps
        row = self.records[i]
        path = ROOT / row['image_path']
        if sha256(path) != row['image_sha256']:
            raise ValueError('Image checksum mismatch.')
        with Image.open(path) as im:
            image = im.convert('RGB')
            if row.get('transform') == 'horizontal_flip':
                image = ImageOps.mirror(image)
            return self.preprocess(image)
