"""Dataset-independent cosine classification; no training or prompt search."""

import csv
import importlib.metadata
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone

from .metrics import classification_metrics
from .io_utils import sha256, write_json, inside_repo, command_output, configure_paths, source_snapshot
from .configuration import read_config
from .model_adapters import load_model
from .dataset_adapters import load_dataset


def run(root, args):
    config = read_config(root, args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = args.run_name or f"{'smoke' if args.limit is not None else 'full'}_{stamp}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_name):
        raise ValueError("Run name must be a short directory name using letters, digits, _, - or .")
    output = inside_repo(root, config["results_dir"]) / run_name
    if output.exists():
        raise FileExistsError(f"Run already exists: {output}. Choose a new --run-name.")
    output.mkdir(parents=True)
    configure_paths(root)
    source_hashes, snapshot = source_snapshot(root, [inside_repo(root, args.config)])
    manifest = {
        "artifact_schema_version": 2, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": sys.argv, "run_name": run_name,
        "git_head": command_output(["git", "rev-parse", "HEAD"], root),
        "git_status": command_output(["git", "status", "--short", "--untracked-files=normal"], root),
        "source_sha256": source_hashes, "source_snapshot": snapshot,
    }
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        evaluate(root, output, config, manifest)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["wall_seconds"] = time.perf_counter() - started
        write_json(output / "manifest.json", manifest)


def evaluate(root, output, config, manifest):
    print(f"Run directory: {output}\nLoading libraries...", flush=True)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; fix the environment or explicitly use --device cpu.")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    write_json(output / "environment.json", {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": dict(sorted(packages.items(), key=lambda pair: pair[0].lower())),
        "torch_cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_driver": command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], root),
        "deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32": False, "cpu_threads": torch.get_num_threads(),
    })

    adapter = load_model(root, config, device)
    bundle = load_dataset(root, config, adapter.preprocess)
    dataset = bundle.images
    class_count = len(bundle.classes)
    embedding_dim = adapter.embedding_dim
    classes = [
        {"class_id": i, "dataset_name": name, "prompt_name": name.replace("_", " "),
         "prompt": config["prompt_template"].format(**{"class": name.replace("_", " ")})}
        for i, name in enumerate(bundle.classes)
    ]
    write_json(output / "classes.json", classes)
    count = config["limit"] if config["limit"] is not None else len(dataset)
    if not 1 <= count <= len(dataset):
        raise ValueError("Requested sample count exceeds the dataset split.")
    samples = [{"test_index": i, "sample_id": bundle.sample_ids[i], "target_id": bundle.targets[i]}
               for i in range(count)]
    write_json(output / "samples.json", samples)
    manifest.update(adapter.metadata)
    manifest.update({
        "dataset": {**bundle.metadata, "evaluated_samples": count},
        "mode": "full_test" if count == len(dataset) else "smoke_test",
    })
    write_json(output / "manifest.json", manifest)
    loader = DataLoader(Subset(dataset, range(count)), batch_size=config["batch_size"],
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda", drop_last=False)

    def normalize_checked(features):
        if not torch.isfinite(features).all() or (features.norm(dim=-1) <= 0).any():
            raise RuntimeError("Non-finite or zero embedding.")
        normalized = torch.nn.functional.normalize(features.float(), dim=-1)
        if not torch.allclose(normalized.norm(dim=-1), torch.ones(features.shape[0], device=device), atol=1e-5):
            raise RuntimeError("Embedding normalization failed.")
        return normalized

    targets, predictions = [], []
    fields = ["sample_id", "test_index", "target_id", "target_name", "prediction_id", "prediction_name",
              "correct", "predicted_cosine", "target_cosine", "top1_top2_margin"]
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    evaluation_started = time.perf_counter()
    with torch.inference_mode(), (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        text_features = normalize_checked(adapter.encode_texts([c["prompt"] for c in classes]))
        if tuple(text_features.shape) != (class_count, embedding_dim):
            raise RuntimeError(f"Unexpected text feature shape: {text_features.shape}")
        print(f"Text embeddings: {tuple(text_features.shape)}; evaluating {count} images...", flush=True)
        for batch_number, (images, labels) in enumerate(loader, start=1):
            start_index = len(targets)
            expected = bundle.targets[start_index:start_index + len(labels)]
            if labels.tolist() != expected:
                raise RuntimeError("DataLoader targets differ from official test index order.")
            images = images.to(device, non_blocking=device.type == "cuda")
            image_features = normalize_checked(adapter.encode_images(images))
            if tuple(image_features.shape) != (len(labels), embedding_dim):
                raise RuntimeError(f"Unexpected image feature shape: {image_features.shape}")
            scores = image_features @ text_features.T
            if not torch.isfinite(scores).all():
                raise RuntimeError("Non-finite cosine scores.")
            # torch.argmax resolves exact ties to the lowest class index.
            predicted = scores.argmax(dim=1)
            top_two = scores.topk(2, dim=1).values
            batch_predictions = predicted.cpu().tolist()
            best = scores.gather(1, predicted[:, None]).squeeze(1).cpu().tolist()
            true_scores = scores.gather(1, labels.to(device)[:, None]).squeeze(1).cpu().tolist()
            margins = (top_two[:, 0] - top_two[:, 1]).cpu().tolist()
            for offset, (target, prediction) in enumerate(zip(expected, batch_predictions)):
                index = start_index + offset
                writer.writerow({
                    "sample_id": bundle.sample_ids[index], "test_index": index,
                    "target_id": target, "target_name": bundle.classes[target],
                    "prediction_id": prediction, "prediction_name": bundle.classes[prediction],
                    "correct": int(target == prediction), "predicted_cosine": best[offset],
                    "target_cosine": true_scores[offset], "top1_top2_margin": margins[offset],
                })
            targets.extend(expected)
            predictions.extend(batch_predictions)
            handle.flush()
            if batch_number == 1 or batch_number % 10 == 0 or len(targets) == count:
                print(f"Evaluated {len(targets)}/{count} images", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    metrics, per_class, confusion = classification_metrics(targets, predictions, bundle.classes)
    metrics["evaluation_seconds"] = time.perf_counter() - evaluation_started
    metrics["images_per_second"] = count / metrics["evaluation_seconds"]
    metrics["peak_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    write_json(output / "metrics.json", metrics)
    with (output / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class_id", "true_class_name", *range(class_count)])
        for i, row in enumerate(confusion):
            writer.writerow([i, bundle.classes[i], *row])
    manifest["embedding_dimensions"] = {"image": embedding_dim, "text": embedding_dim, "text_classes": class_count}
    manifest["artifact_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != "manifest.json"}
    print(f"Top-1: {metrics['correct']}/{count} = {metrics['top1_accuracy_percent']:.4f}%", flush=True)
    print(f"Artifacts saved to: {output}", flush=True)
