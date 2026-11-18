"""OpenAI CLIP checkpoint, implemented with OpenCLIP and QuickGELU."""

from ..io_utils import inside_repo, sha256

CHECKPOINT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
CHECKPOINT_URL = f"https://openaipublic.azureedge.net/clip/models/{CHECKPOINT_SHA256}/ViT-B-32.pt"


def validate_config(config):
    for key, value in {"model": "ViT-B-32", "pretrained": "openai",
                       "checkpoint_sha256": CHECKPOINT_SHA256,
                       "checkpoint_url": CHECKPOINT_URL}.items():
        if config[key] != value:
            raise ValueError(f"OpenAI CLIP adapter requires {key}={value!r}.")


class OpenAIClip:
    def __init__(self, root, config, device):
        import open_clip
        import torch

        validate_config(config)
        self.device = device
        checkpoint = inside_repo(root, config["model_dir"]) / "ViT-B-32.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists():
            print("Downloading official pretrained checkpoint...", flush=True)
            torch.hub.download_url_to_file(config["checkpoint_url"], str(checkpoint),
                                          hash_prefix=config["checkpoint_sha256"], progress=True)
        actual_hash = sha256(checkpoint)
        if actual_hash != config["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint SHA256 mismatch: {checkpoint}")
        pretrained_cfg = open_clip.get_pretrained_cfg(config["model"], config["pretrained"])
        print("Checkpoint verified. Loading pretrained CLIP with QuickGELU...", flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config["model"], pretrained=str(checkpoint), device=device, precision=config["precision"],
            force_quick_gelu=True, image_mean=pretrained_cfg["mean"], image_std=pretrained_cfg["std"],
            image_interpolation=pretrained_cfg["interpolation"], image_resize_mode=pretrained_cfg["resize_mode"],
            # Official TorchScript archive, loaded only after verifying its pinned SHA256.
            weights_only=False, cache_dir=str(checkpoint.parent),
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(config["model"])
        self.embedding_dim = int(self.model.text_projection.shape[-1])
        if self.model.visual.output_dim != self.embedding_dim:
            raise RuntimeError("Model image/text embedding dimensions differ.")
        self.metadata = {
            "model_adapter": "openai_clip", "model_embedding_dim": self.embedding_dim,
            "checkpoint": {"url": config["checkpoint_url"], "sha256": actual_hash,
                           "path": checkpoint.relative_to(root).as_posix(), "quick_gelu": True},
            "preprocessing": repr(self.preprocess), "pretrained_config": pretrained_cfg,
            "tokenizer": type(self.tokenizer).__name__, "context_length": self.model.context_length,
        }

    def encode_texts(self, texts):
        return self.model.encode_text(self.tokenizer(texts).to(self.device))

    def encode_images(self, images):
        return self.model.encode_image(images)
