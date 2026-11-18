"""Filesystem and provenance utilities shared by runners and audits."""

import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_snapshot(root, extra_paths=()):
    paths = [*(root / "vlm_probe").rglob("*.py"), root / "requirements.txt", *extra_paths]
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(paths))}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:20]
    path = root / "results/source_snapshots" / f"source_{identity}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
            for name in hashes:
                archive.write(root / name, name)
    with zipfile.ZipFile(path) as archive:
        if any(hashlib.sha256(archive.read(name)).hexdigest() != digest for name, digest in hashes.items()):
            raise ValueError("Source snapshot does not match executable source.")
    return hashes, {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}


def inside_repo(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path must be inside the project folder: {value}")
    return path


def command_output(command, root):
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=15, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def configure_paths(root):
    cache = root / ".cache"
    for variable, subdir in {
        "HF_HOME": "huggingface", "TORCH_HOME": "torch", "XDG_CACHE_HOME": "xdg",
        "MPLCONFIGDIR": "matplotlib", "TEMP": "tmp", "TMP": "tmp", "TMPDIR": "tmp",
    }.items():
        path = cache / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


