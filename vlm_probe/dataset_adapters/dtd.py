"""DTD primary texture categories, test partition 1 only."""

from . import DatasetBundle
from .archive_utils import prepare_archive, verify_members, validate_split, provenance
from ..io_utils import inside_repo, sha256

ORIGIN = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
MIRROR = "https://huggingface.co/datasets/xingslong/dtd/resolve/2afd1ec578464e8d2423a3c0cff74353df43d924/dtd-r1.0.1.tar.gz"
MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"
PROTOCOL = {"split": "test", "partition": 1, "num_classes": 47, "num_samples": 1880,
            "samples_per_class": 40, "namespace": "dtd/test/p1", "download_urls": [ORIGIN, MIRROR],
            "prompt_template": "a photo of a {class} texture.",
            "sample_id_pattern": r"dtd/test/p1/[a-z-]+/[A-Za-z0-9_-]+\.jpg"}


def load(root, config, preprocess):
    from torchvision.datasets import DTD

    base = inside_repo(root, config["data_dir"]) / "dtd"
    archive = base / "dtd-r1.0.1.tar.gz"
    prepare_archive(archive, [config["dataset_download_url"]], MD5, base, base / "dtd/labels/test1.txt")
    metadata = [base / "dtd/labels" / f"{split}1.txt" for split in ("train", "val", "test")]
    parts = [p.read_text().splitlines() for p in metadata]
    if any(len(x) != 1880 for x in parts) or len(set(sum(parts, []))) != 5640:
        raise ValueError("DTD partition 1 split sizes/disjointness invalid.")
    paths = [base / "dtd/images" / name for name in parts[2]]
    verified = verify_members(archive, base, [*metadata, *paths])
    dataset = DTD(str(base.parent), split="test", partition=1, transform=preprocess, download=False)
    classes = sorted({name.split("/")[0] for name in parts[2]})
    targets = [classes.index(name.split("/")[0]) for name in parts[2]]
    if dataset.classes != classes or dataset._labels != targets or dataset._image_files != paths:
        raise ValueError("DTD loader differs from original partition metadata.")
    return validate_split(DatasetBundle(dataset, targets, classes,
        [f"dtd/test/p1/{name}" for name in parts[2]],
        {"name": "DTD", "split": "test", "partition": 1, "origin_url": ORIGIN,
         "download_url": config["dataset_download_url"], "archive_md5": MD5,
         "archive": provenance(archive), "metadata_sha256": {p.name: sha256(p) for p in metadata},
         "verified_files_digest": verified, "test_images_archive_verified": True,
         "torchvision_integrity_checked": True, "sample_order": "labels/test1.txt"}), PROTOCOL)
