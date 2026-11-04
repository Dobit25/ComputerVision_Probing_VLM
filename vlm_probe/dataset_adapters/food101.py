"""Food-101 original test split: 250 images in each of 101 classes."""

import json

from . import DatasetBundle
from .archive_utils import prepare_archive, verify_members, validate_split, provenance
from ..io_utils import inside_repo, sha256

URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
MIRROR = "https://huggingface.co/datasets/xingslong/food-101/resolve/716b18d35b1af8868010e682dc1bf2634fc3afb0/food-101.tar.gz?download=true"
MD5 = "85eeb15f3717b99a5da872d97d918f87"
PROTOCOL = {"split": "test", "num_classes": 101, "num_samples": 25250,
            "samples_per_class": 250, "namespace": "food101/test", "download_urls": [URL, MIRROR],
            "sample_id_pattern": r"food101/test/[a-z_]+/[0-9]+\.jpg"}


def load(root, config, preprocess):
    from torchvision.datasets import Food101

    data = inside_repo(root, config["data_dir"])
    base = data / "food-101"
    archive = data / "food-101.tar.gz"
    prepare_archive(archive, [config["dataset_download_url"]], MD5, data, base / "meta/test.json")
    metadata = [base / "meta" / name for name in ("test.json", "train.json", "classes.txt")]
    test, train = [json.loads(p.read_text()) for p in metadata[:2]]
    test_ids = [name for names in test.values() for name in names]
    train_ids = [name for names in train.values() for name in names]
    if len(test_ids) != 25250 or len(train_ids) != 75750 or len(set(test_ids + train_ids)) != 101000:
        raise ValueError("Food-101 split sizes, uniqueness or disjointness invalid.")
    paths = [base / "images" / f"{name}.jpg" for name in test_ids]
    verified = verify_members(archive, data, [*metadata, *paths])
    dataset = Food101(str(data), split="test", transform=preprocess, download=False)
    classes = sorted(test)
    targets = [classes.index(name.split("/")[0]) for name in test_ids]
    if dataset.classes != classes or dataset._labels != targets or dataset._image_files != paths:
        raise ValueError("Food-101 loader differs from original metadata.")
    return validate_split(DatasetBundle(dataset, targets, classes,
        [f"food101/test/{name}.jpg" for name in test_ids],
        {"name": "Food101", "split": "test", "origin_url": URL,
         "download_url": config["dataset_download_url"], "archive_md5": MD5,
         "archive": provenance(archive), "metadata_sha256": {p.name: sha256(p) for p in metadata},
         "verified_files_digest": verified, "test_images_archive_verified": True,
         "torchvision_integrity_checked": True, "sample_order": "meta/test.json"}), PROTOCOL)
