"""Dataset contracts and lazy registry; importing this module needs no ML packages."""

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetBundle:
    images: Any
    targets: list[int]
    classes: list[str]
    sample_ids: list[str]
    metadata: dict

    def validate(self):
        size = len(self.images)
        if not size or len(self.targets) != size or len(self.sample_ids) != size:
            raise ValueError("Dataset images, labels and IDs must have equal nonzero lengths.")
        if len(set(self.sample_ids)) != size or len(set(self.classes)) != len(self.classes):
            raise ValueError("Duplicate sample IDs or class names.")
        if len(self.classes) < 2 or any(type(x) is not int or not 0 <= x < len(self.classes) for x in self.targets):
            raise ValueError("Invalid class mapping or labels.")
        support = Counter(self.targets)
        self.metadata.update(num_classes=len(self.classes), total_test_samples=size,
                             full_split_support=[support[i] for i in range(len(self.classes))])
        return self


def get_protocol(name):
    if name in ("OxfordIIITPet", "DTD", "Food101"):
        from importlib import import_module
        module = {"OxfordIIITPet": "oxford_pet", "DTD": "dtd", "Food101": "food101"}[name]
        return import_module(f"{__name__}.{module}").PROTOCOL
    if name == "CIFAR100":
        from .cifar100 import PROTOCOL
        return PROTOCOL
    if name == "Flowers102":
        from .flowers102 import PROTOCOL
        return PROTOCOL
    raise ValueError(f"No dataset adapter registered for {name!r}.")


def validate_dataset_config(config):
    protocol = get_protocol(config["dataset"])
    if config["split"] != protocol["split"]:
        raise ValueError("Unsupported dataset split.")
    if config["dataset_download_url"] not in protocol["download_urls"]:
        raise ValueError("Use an approved dataset source URL.")
    return protocol


def load_dataset(root, config, preprocess):
    validate_dataset_config(config)
    if config["dataset"] in ("OxfordIIITPet", "DTD", "Food101"):
        from importlib import import_module
        module = {"OxfordIIITPet": "oxford_pet", "DTD": "dtd", "Food101": "food101"}[config["dataset"]]
        load = import_module(f"{__name__}.{module}").load
    elif config["dataset"] == "CIFAR100":
        from .cifar100 import load
    elif config["dataset"] == "Flowers102":
        from .flowers102 import load
    else:
        raise ValueError("Dataset adapter is not registered.")
    return load(root, config, preprocess).validate()
