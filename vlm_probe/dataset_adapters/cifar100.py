"""CIFAR-100 fine-label official test protocol and data loader."""

from pathlib import Path

from . import DatasetBundle
from ..io_utils import inside_repo, sha256

PROTOCOL = {
    "split": "test", "num_classes": 100, "num_samples": 10000,
    "namespace": "cifar100/test", "samples_per_class": 100,
    "sample_id_template": "cifar100/test/{index:05d}",
    "download_urls": ["https://dataset.bj.bcebos.com/cifar/cifar-100-python.tar.gz",
                      "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"],
}


def load(root, config, preprocess):
    from torchvision.datasets import CIFAR100

    class DownloadableCIFAR100(CIFAR100):
        url = config["dataset_download_url"]

    print("Preparing checksum-verified CIFAR-100 test split...", flush=True)
    dataset = DownloadableCIFAR100(root=str(inside_repo(root, config["data_dir"])), train=False,
                                   transform=preprocess, download=True)
    if len(dataset) != PROTOCOL["num_samples"] or len(dataset.classes) != PROTOCOL["num_classes"]:
        raise RuntimeError("Unexpected CIFAR-100 test size or class count.")
    if dataset.class_to_idx != {name: i for i, name in enumerate(dataset.classes)}:
        raise RuntimeError("Dataset class order does not match target IDs.")
    if sorted(dataset.targets) != [i for i in range(100) for _ in range(100)]:
        raise RuntimeError("Expected exactly 100 test images per fine class.")
    data_root = Path(dataset.root) / dataset.base_folder
    return DatasetBundle(
        dataset, list(dataset.targets), list(dataset.classes),
        [f"cifar100/test/{i:05d}" for i in range(len(dataset))],
        {"name": "CIFAR100", "split": "test", "origin_url": CIFAR100.url,
         "download_url": dataset.url, "archive_md5": dataset.tgz_md5,
         "test_sha256": sha256(data_root / "test"), "meta_sha256": sha256(data_root / "meta"),
         "torchvision_integrity_checked": True},
    )
