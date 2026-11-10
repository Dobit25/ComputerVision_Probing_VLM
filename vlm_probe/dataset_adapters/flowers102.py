"""Oxford Flowers102: original tstid split, zero-based labels, source image IDs."""

import hashlib
import importlib.metadata
import tarfile

from . import DatasetBundle
from ..io_utils import inside_repo, sha256

ORIGIN = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
MIRROR = "https://paddlemodels.bj.bcebos.com/flowers/"
FILES = {
    "102flowers.tgz": "52808999861908f626f3c1f4e79d11fa",
    "imagelabels.mat": "e0620be6f572b9609742df49c70aed4d",
    "setid.mat": "a5357ecc9cb78c4bef273ce3793fc85c",
}
PROTOCOL = {"split": "test", "num_classes": 102, "num_samples": 6149,
            "namespace": "flowers102/test", "download_urls": [ORIGIN, MIRROR],
            "sample_id_pattern": r"flowers102/test/image_[0-9]{5}\.jpg"}


def verify_test_images(archive_path, image_paths):
    """Check extracted test images against the checksum-verified original archive."""
    expected = {f"jpg/{p.name}": p for p in image_paths}
    if len(expected) != len(image_paths):
        raise RuntimeError("Duplicate Flowers102 image IDs.")
    digests = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name not in expected:
                continue
            if not member.isfile() or member.name in digests:
                raise RuntimeError("Invalid or duplicate archive image entry.")
            with archive.extractfile(member) as stream:
                if hasattr(hashlib, "file_digest"):
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                else:
                    digest = hashlib.sha256(stream.read()).hexdigest()
            if sha256(expected[member.name]) != digest:
                raise RuntimeError(f"Extracted image differs from original archive: {member.name}")
            digests[member.name] = digest
    if set(digests) != set(expected):
        raise RuntimeError("Test images missing from official archive.")
    # Stable digest of every test filename and its bytes, independent of smoke limit.
    canonical = "".join(f"{name}\t{digests[name]}\n" for name in sorted(digests))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load(root, config, preprocess):
    import numpy as np
    from scipy.io import loadmat
    from torchvision.datasets import Flowers102
    from torchvision.datasets.utils import download_url

    base = inside_repo(root, config["data_dir"]) / "flowers-102"
    base.mkdir(parents=True, exist_ok=True)
    print("Preparing Flowers102: verifying archive, labels and official split...", flush=True)
    # Always check all three files, even when torchvision finds an existing jpg folder.
    for name, md5 in FILES.items():
        download_url(config["dataset_download_url"] + name, str(base), filename=name, md5=md5)

    class DownloadableFlowers102(Flowers102):
        _download_url_prefix = config["dataset_download_url"]

    dataset = DownloadableFlowers102(root=str(inside_repo(root, config["data_dir"])), split="test",
                                     transform=preprocess, download=True)
    split_mat = loadmat(base / "setid.mat", squeeze_me=True)
    splits = {key: [int(x) for x in np.atleast_1d(split_mat[key])] for key in ("trnid", "valid", "tstid")}
    if [len(splits[k]) for k in ("trnid", "valid", "tstid")] != [1020, 1020, PROTOCOL["num_samples"]]:
        raise RuntimeError("Unexpected Flowers102 official split sizes.")
    all_ids = [x for ids in splits.values() for x in ids]
    if sorted(all_ids) != list(range(1, 8190)):
        raise RuntimeError("Flower splits overlap or do not cover the original image IDs.")
    labels = [int(x) for x in loadmat(base / "imagelabels.mat", squeeze_me=True)["labels"]]
    image_ids = splits["tstid"]
    targets = [labels[i - 1] - 1 for i in image_ids]
    image_paths = [base / "jpg" / f"image_{i:05d}.jpg" for i in image_ids]
    if len(dataset) != PROTOCOL["num_samples"] or len(dataset.classes) != PROTOCOL["num_classes"]:
        raise RuntimeError("Unexpected Flowers102 test size or class count.")
    if dataset._labels != targets or list(dataset._image_files) != image_paths:
        raise RuntimeError("Torchvision labels/image order differ from the official tstid protocol.")
    if set(targets) != set(range(PROTOCOL["num_classes"])):
        raise RuntimeError("Test split does not cover all Flowers102 classes.")
    print("Checking extracted test image bytes against the original archive...", flush=True)
    images_hash = verify_test_images(base / "102flowers.tgz", image_paths)
    return DatasetBundle(
        dataset, targets, list(dataset.classes),
        [f"flowers102/test/image_{i:05d}.jpg" for i in image_ids],
        {"name": "Flowers102", "split": "test", "split_key": "tstid", "origin_url": ORIGIN,
         "download_url": config["dataset_download_url"], "source_md5": FILES,
         "source_sha256": {name: sha256(base / name) for name in FILES},
         "test_images_sha256": images_hash, "test_images_archive_verified": True,
         "label_policy": "original_1_based_minus_1", "sample_order": "setid.mat:tstid",
         "class_names_source": f"torchvision.datasets.Flowers102.classes@{importlib.metadata.version('torchvision')}",
         "class_names_sha256": hashlib.sha256("\n".join(dataset.classes).encode("utf-8")).hexdigest(),
         "torchvision_integrity_checked": True},
    )
