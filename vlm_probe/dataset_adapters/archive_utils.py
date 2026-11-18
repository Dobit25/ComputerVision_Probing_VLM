"""Integrity checks shared by datasets distributed as original tar archives."""

import hashlib
import json
import tarfile
from collections import Counter

from ..assets import obtain
from ..io_utils import sha256


def prepare_archive(path, urls, md5, destination, marker):
    obtain(path, urls, md5, "md5")
    if not marker.exists():
        print(f"Extracting {path.name}...", flush=True)
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(destination, filter="data")


def verify_members(archive_path, extraction_root, paths):
    expected = {p.relative_to(extraction_root).as_posix(): p for p in paths}
    if len(expected) != len(paths):
        raise ValueError("Duplicate paths in dataset input.")
    actual = {}
    print(f"Checking {len(paths)} extracted files against {archive_path.name}...", flush=True)
    # Stream the archive first, then read extracted files in filename order.
    # Alternating archive and image reads causes excessive seeks on slower disks.
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in expected:
                continue
            if not member.isfile() or name in actual:
                raise ValueError(f"Invalid archive entry: {name}")
            with archive.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            actual[name] = digest
            if len(actual) % 5000 == 0:
                print(f"Archive members hashed: {len(actual)}/{len(expected)}", flush=True)
    if set(expected) != set(actual):
        raise ValueError("Missing dataset files in archive.")
    for i, name in enumerate(sorted(expected), 1):
        if sha256(expected[name]) != actual[name]:
            raise ValueError(f"Extracted file differs from verified archive: {name}")
        if i % 5000 == 0:
            print(f"Extracted files verified: {i}/{len(expected)}", flush=True)
    canonical = "".join(f"{name}\t{actual[name]}\n" for name in sorted(actual))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_split(bundle, protocol):
    bundle.validate()
    if len(bundle.images) != protocol["num_samples"] or len(bundle.classes) != protocol["num_classes"]:
        raise ValueError("Dataset shape differs from official protocol.")
    if set(bundle.targets) != set(range(len(bundle.classes))):
        raise ValueError("Full split is missing a class.")
    count = protocol.get("samples_per_class")
    if count is not None and set(Counter(bundle.targets).values()) != {count}:
        raise ValueError("Official class support mismatch.")
    return bundle


def provenance(path):
    sidecar = path.with_name(path.name + ".source.json")
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size,
            "download": json.loads(sidecar.read_text()) if sidecar.exists() else None}
