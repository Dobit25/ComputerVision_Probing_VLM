"""Registry for dual-encoder models used by the cosine classifier."""

from typing import Any, Protocol


class ModelAdapter(Protocol):
    preprocess: Any
    embedding_dim: int
    metadata: dict

    def encode_images(self, images: Any) -> Any: ...
    def encode_texts(self, texts: list[str]) -> Any: ...


def validate_model_config(config):
    from .openai_clip import validate_config
    validate_config(config)


def load_model(root, config, device) -> ModelAdapter:
    from .openai_clip import OpenAIClip
    return OpenAIClip(root, config, device)
