"""Original Oxford-IIIT Pet breed classification, official test split."""

from . import DatasetBundle
from .archive_utils import prepare_archive, verify_members, validate_split, provenance
from ..io_utils import inside_repo, sha256

ORIGIN = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
MIRROR = "https://huggingface.co/datasets/xingslong/oxford-iiit-pet/resolve/e6b0d43e87d842f13fc4073b321e840c4b10f943/"
PROTOCOL = {"split": "test", "num_classes": 37, "num_samples": 3669,
            "namespace": "oxford_pet/test", "download_urls": [ORIGIN, MIRROR],
            "sample_id_pattern": r"oxford_pet/test/[A-Za-z_]+_[0-9]+\.jpg"}
FILES = {"images.tar.gz": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
         "annotations.tar.gz": "95a8c909bbe2e81eed6a22bccdf3f68f"}


def load(root, config, preprocess):
    from torchvision.datasets import OxfordIIITPet

    base = inside_repo(root, config["data_dir"]) / "oxford-iiit-pet"
    for filename, md5 in FILES.items():
        marker = base / ("images" if filename == "images.tar.gz" else "annotations/test.txt")
        prepare_archive(base / filename, [config["dataset_download_url"] + filename], md5, base, marker)
    metadata = [base / "annotations" / name for name in ("test.txt", "trainval.txt", "list.txt")]
    meta_hash = verify_members(base / "annotations.tar.gz", base, metadata)
    test = [line.split() for line in metadata[0].read_text().splitlines() if line.strip()]
    train = [line.split()[0] for line in metadata[1].read_text().splitlines() if line.strip()]
    ids = [row[0] for row in test]
    if len(train) != 3680 or len(set(train)) != 3680 or len(set(ids)) != 3669 or set(ids) & set(train):
        raise ValueError("Pet split sizes, uniqueness or disjointness invalid.")
    dataset = OxfordIIITPet(str(base.parent), split="test", target_types="category",
                            transform=preprocess, download=False)
    targets = [int(row[1]) - 1 for row in test]
    paths = [base / "images" / f"{name}.jpg" for name in ids]
    if dataset._labels != targets or dataset._images != paths:
        raise ValueError("Pet loader differs from original metadata.")
    images_hash = verify_members(base / "images.tar.gz", base, paths)
    return validate_split(DatasetBundle(dataset, targets, list(dataset.classes),
        [f"oxford_pet/test/{name}.jpg" for name in ids],
        {"name": "OxfordIIITPet", "split": "test", "target_types": "category", "origin_url": ORIGIN,
         "download_url": config["dataset_download_url"], "source_md5": FILES,
         "archives": {name: provenance(base / name) for name in FILES},
         "metadata_sha256": {p.name: sha256(p) for p in metadata}, "verified_metadata_digest": meta_hash,
         "test_images_sha256": images_hash, "test_images_archive_verified": True,
         "torchvision_integrity_checked": True, "sample_order": "annotations/test.txt"}), PROTOCOL)
