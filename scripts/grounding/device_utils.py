"""Shared device / model loading helpers for local and server eval."""

from __future__ import annotations

import os

import torch


def get_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_local_model_dir(model_dir: str) -> bool:
    model_dir = os.path.expanduser(model_dir)
    return os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "config.json"))


def load_hf_model(ModelCls, ProcessorCls, model_dir: str, hf_model_id: str):
    """Load from local cache or download from Hugging Face and cache under model_dir."""
    model_dir = os.path.expanduser(model_dir)
    if is_local_model_dir(model_dir):
        print(f"Loading from local cache: {model_dir}")
        processor = ProcessorCls.from_pretrained(model_dir, local_files_only=True)
        model = ModelCls.from_pretrained(model_dir, local_files_only=True)
    else:
        print(f"Local model missing at {model_dir}; downloading {hf_model_id} ...")
        os.makedirs(model_dir, exist_ok=True)
        processor = ProcessorCls.from_pretrained(hf_model_id)
        model = ModelCls.from_pretrained(hf_model_id)
        processor.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        print(f"Cached model to {model_dir}")
    return model, processor
