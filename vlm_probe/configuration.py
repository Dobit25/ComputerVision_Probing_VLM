"""Validate experiment protocols before downloading or creating output files."""

import json

from .dataset_adapters import validate_dataset_config
from .io_utils import inside_repo
from .model_adapters import validate_model_config

CONFIG_KEYS = {
    "protocol_version", "model", "pretrained", "checkpoint_url", "checkpoint_sha256",
    "dataset", "dataset_download_url", "split", "prompt_template", "class_name_policy", "precision", "device",
    "batch_size", "num_workers", "seed", "cpu_threads", "data_dir", "model_dir", "results_dir",
}


def validate_config(config, effective=False):
    required = CONFIG_KEYS | ({"limit", "selection"} if effective else set())
    if set(config) != required:
        raise ValueError(f"Config keys differ from schema: {set(config) ^ required}")
    if type(config["protocol_version"]) is not int or config["protocol_version"] not in (1, 2, 3):
        raise ValueError("Unsupported experiment protocol version.")
    if config["protocol_version"] == 1 and config["dataset"] != "CIFAR100":
        raise ValueError("Protocol v1 fixes dataset=CIFAR100.")
    if config["protocol_version"] < 3 and config["dataset"] not in ("CIFAR100", "Flowers102"):
        raise ValueError("New datasets require protocol v3.")
    protocol = validate_dataset_config(config)
    for key, value in {"prompt_template": protocol.get("prompt_template", "a photo of a {class}."),
                       "class_name_policy": "replace_underscores_with_spaces",
                       "precision": "fp32", "num_workers": 0}.items():
        if config[key] != value or type(config[key]) is not type(value):
            raise ValueError(f"Protocol fixes {key}={value!r}.")
    validate_model_config(config)
    for key in ("batch_size", "cpu_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    if effective:
        limit = config["limit"]
        if limit is not None and (type(limit) is not int or not 1 <= limit <= protocol["num_samples"]):
            raise ValueError(f"--limit must be between 1 and {protocol['num_samples']}.")
        if config["selection"] != ("all_test_indices" if limit is None else "first_n_test_indices"):
            raise ValueError("Selection does not match limit.")
    return protocol


def read_config(root, args):
    config = json.loads(inside_repo(root, args.config).read_text(encoding="utf-8"))
    validate_config(config)
    for key in ("batch_size", "device"):
        override = getattr(args, key)
        if override is not None:
            config[key] = override
    config["limit"] = args.limit
    config["selection"] = "all_test_indices" if args.limit is None else "first_n_test_indices"
    validate_config(config, effective=True)
    for key in ("data_dir", "model_dir", "results_dir"):
        inside_repo(root, config[key])
    return config
