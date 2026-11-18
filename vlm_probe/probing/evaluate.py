"""Run the preregistered one-model probing suite and retain every caption score."""

import importlib.metadata
import json
import os
import pickle
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..assets import digest_file
from ..configuration import validate_config
from ..dataset_adapters import load_dataset
from ..io_utils import configure_paths, inside_repo, sha256, source_snapshot, write_json
from ..model_adapters import load_model
from .metrics import score_row, summarize
from .stimuli import materialize, ordered_samples


def fine_samples(root, bundle):
    data_root = root / "data/cifar-100-python"
    if digest_file(data_root / "test", "md5") != "f0ef6b0ae62326f3e7ffdfab6717acfc":
        raise ValueError("CIFAR test checksum differs from the official archive.")
    if digest_file(data_root / "meta", "md5") != "7973b15100ade9c7d40fb424638fde48":
        raise ValueError("CIFAR meta checksum mismatch.")
    with (data_root / "test").open("rb") as stream:
        test = pickle.load(stream, encoding="latin1")
    with (data_root / "meta").open("rb") as stream:
        meta = pickle.load(stream, encoding="latin1")
    if test["fine_labels"] != bundle.targets:
        raise ValueError("Fine-grained sample order does not match classifier.")
    mapping = {}
    for fine, coarse in zip(test["fine_labels"], test["coarse_labels"]):
        if fine in mapping and mapping[fine] != coarse:
            raise ValueError("Inconsistent fine/coarse mapping.")
        mapping[fine] = coarse
    groups = {coarse: sorted(fine for fine, c in mapping.items() if c == coarse) for coarse in range(20)}
    if any(len(group) != 5 for group in groups.values()):
        raise ValueError("Expected 20 official coarse groups, each with five fine classes.")
    rows = []
    for i, (fine, coarse) in enumerate(zip(test["fine_labels"], test["coarse_labels"])):
        candidates = groups[coarse]
        rows.append({"sample_id": f"fine_grained/cifar100/test/{i:05d}", "group_id": f"cifar100/{i}",
                     "task": "fine_grained", "condition": meta["coarse_label_names"][coarse],
                     "stratum": meta["coarse_label_names"][coarse], "target": candidates.index(fine),
                     "captions": [f"a photo of a {bundle.classes[c].replace('_', ' ')}." for c in candidates],
                     "candidate_class_ids": candidates, "global_target_id": fine, "test_index": i,
                     "coarse_id": coarse, "image_source": bundle.sample_ids[i]})
    return rows


def run(root, config_path, run_name, smoke=False):
    import re

    config = json.loads(inside_repo(root, config_path).read_text())
    expected = {"suite_version": 1, "renderer_seed": 14, "inference_seed": 0,
                "batch_size": 32, "text_batch_size": 32, "bootstrap_draws": 1000, "bootstrap_seed": 0,
                "model_config": "configs/clip_b32_cifar100.json", "baseline_run": "results/full_cifar100_clip_b32_v1",
                "tasks": ["counting", "spatial", "negation", "fine_grained"],
                "expected_samples": {"counting": 1080, "spatial": 720, "negation": 1440, "fine_grained": 10000}}
    if config != expected:
        raise ValueError("Probe config differs from preregistered v1 protocol.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_name):
        raise ValueError("Invalid run name.")
    output = root / "results" / run_name
    if output.exists():
        raise FileExistsError(output)
    configure_paths(root)
    output.mkdir()
    hashes, snapshot = source_snapshot(root, [root / config_path, root / config["model_config"], root / "docs/probing_protocol.md"])
    manifest = {"status": "running", "kind": "core_probing_v1", "mode": "smoke" if smoke else "full",
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "command_argv": sys.argv,
                "source_sha256": hashes, "source_snapshot": snapshot}
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        evaluate(root, output, config, manifest, smoke)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)


def evaluate(root, output, config, manifest, smoke):
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    print(f"Probe output: {output}", flush=True)
    model_config = json.loads((root / config["model_config"]).read_text())
    validate_config(model_config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required by this frozen protocol.")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    write_json(output / "environment.json", {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": dict(sorted(packages.items())),
        "device": "cuda", "gpu": torch.cuda.get_device_name(0), "torch_cuda_build": torch.version.cuda,
        "precision": "fp32", "tf32": False, "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "cpu_threads": 4})
    adapter = load_model(root, model_config, torch.device("cuda"))
    manifest.update(adapter.metadata)
    bundle = load_dataset(root, model_config, adapter.preprocess)
    manifest["natural_dataset"] = bundle.metadata
    samples = materialize(root, config["renderer_seed"]) + fine_samples(root, bundle)
    for task, count in config["expected_samples"].items():
        if sum(s["task"] == task for s in samples) != count:
            raise ValueError("Probe dataset size mismatch.")
    for task, count in {"counting": 1080, "spatial": 720, "negation": 720}.items():
        if len({s["image_sha256"] for s in samples if s["task"] == task}) != count:
            raise ValueError("Unexpected duplicate or missing synthetic images.")
    samples = ordered_samples(samples, config["tasks"], 64 if smoke else None)
    write_json(output / "samples.json", samples)

    class ProbeImages(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            if row["task"] == "fine_grained":
                tensor, label = bundle.images[row["test_index"]]
                if label != row["global_target_id"]:
                    raise ValueError("Natural image label mismatch.")
                return tensor
            path = inside_repo(root, row["image_path"])
            if sha256(path) != row["image_sha256"]:
                raise ValueError("Probe image bytes changed.")
            with Image.open(path) as image:
                return adapter.preprocess(image.convert("RGB"))

    def normalize(features):
        if features.ndim != 2 or features.shape[1] != adapter.embedding_dim:
            raise ValueError("Invalid probe embedding shape.")
        if not torch.isfinite(features).all() or (features.norm(dim=-1) <= 0).any():
            raise ValueError("Non-finite/zero probe embeddings.")
        return torch.nn.functional.normalize(features.float(), dim=-1)

    predictions = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for task in config["tasks"]:
            rows = [s for s in samples if s["task"] == task]
            if task == "fine_grained":
                captions = [f"a photo of a {name.replace('_', ' ')}." for name in bundle.classes]
                text = normalize(adapter.encode_texts(captions))  # Same all-100 text batch as historical baseline.
            else:
                captions = list(dict.fromkeys(c for row in rows for c in row["captions"] + row.get("control_captions", [])))
                text = torch.cat([normalize(adapter.encode_texts(captions[i:i + config["text_batch_size"]]))
                                  for i in range(0, len(captions), config["text_batch_size"])])
            text_index = {caption: i for i, caption in enumerate(captions)}
            loader = DataLoader(ProbeImages(rows), batch_size=config["batch_size"], shuffle=False,
                                num_workers=0, pin_memory=True, drop_last=False)
            offset = 0
            for images in loader:
                features = normalize(adapter.encode_images(images.to("cuda", non_blocking=True)))
                all_scores = (features @ text.T).cpu().tolist()
                for sample, scores in zip(rows[offset:offset + len(images)], all_scores):
                    candidates = [scores[text_index[c]] for c in sample["captions"]]
                    controls = [scores[text_index[c]] for c in sample.get("control_captions", [])]
                    global_id = max(range(len(scores)), key=scores.__getitem__) if task == "fine_grained" else None
                    record = score_row(sample, candidates, controls or None, global_id)
                    predictions.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                offset += len(images)
                if offset % 640 == 0 or offset == len(rows):
                    print(f"{task}: {offset}/{len(rows)}", flush=True)
                handle.flush()
    torch.cuda.synchronize()
    manifest["evaluation_seconds"] = time.perf_counter() - started
    manifest["peak_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    manifest["embedding_dim"] = adapter.embedding_dim
    manifest["num_samples"] = len(samples)
    manifest["synthetic_unique_images"] = {task: len({s["image_sha256"] for s in samples if s["task"] == task})
                                             for task in ("counting", "spatial", "negation")}
    manifest["baseline_manifest_sha256"] = sha256(root / config["baseline_run"] / "manifest.json")
    metrics = summarize(predictions)
    write_json(output / "metrics.json", metrics)
    manifest["artifact_sha256"] = {p.name: sha256(p) for p in output.iterdir() if p.name != "manifest.json"}
    print({task: round(stats["accuracy_percent"], 4) for task, stats in metrics.items()}, flush=True)
