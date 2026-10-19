"""Model configurations and device selection.

The dictionaries here are intentionally plain dicts (not dataclasses) so they stay
drop-in compatible with the notebook implementation they were extracted from.
"""

from __future__ import annotations

import torch

# Sizes that OpenAI actually published checkpoints for.
ALLOWED_MODEL_SIZES = ("124M", "355M", "774M", "1558M")

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of transformer blocks
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

# A much smaller variant with the same architecture, used for fast tests, for the
# "scratch-10ep" demo model, and anywhere a full 124M forward pass is wasteful.
# Vocabulary and tokenizer are identical to GPT-2 so traces stay comparable.
GPT_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": True,
}

# Default hyperparameters for the from-scratch training run (see scripts/train_small.py).
SMALL_RUN_CONFIG = {
    "model_config": GPT_CONFIG_SMALL,
    "batch_size": 2,
    "max_length": 256,
    "stride": 256,
    "num_epochs": 10,
    "learning_rate": 0.0004,
    "weight_decay": 0.1,
    "eval_freq": 5,
    "eval_iter": 3,
    "seed": 123,
}


def get_device(preference: str | torch.device | None = None) -> torch.device:
    """Resolve a torch device, preferring CUDA when it is available.

    Replaces the hardcoded ``torch.device("cpu")`` from the notebook. Callers can
    force a device by passing ``preference``; the string ``"auto"`` (or an empty
    string) means "use the best available", which is what the app's settings
    layer passes.
    """
    if preference is not None:
        if isinstance(preference, torch.device):
            return preference
        text = str(preference).strip()
        if text and text.lower() != "auto":
            return torch.device(text)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human readable device name, safe to call for non-CUDA devices."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type