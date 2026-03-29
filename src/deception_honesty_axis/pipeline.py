from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig


def role_file_path(config: ExperimentConfig, source_name: str) -> Path:
    return config.repo_root / "data" / "roles" / "instructions" / source_name


def load_model_and_tokenizer(config: ExperimentConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = str(config.raw["model"].get("dtype", "float16"))
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=bool(config.raw["model"].get("trust_remote_code", False)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = config.raw["model"].get("device", "auto")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=bool(config.raw["model"].get("trust_remote_code", False)),
    )
    model.eval()
    return model, tokenizer


def model_input_device(model):
    return next(model.parameters()).device


def build_prompt_text(tokenizer, system_prompt: str, question_text: str) -> str:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question_text})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if system_prompt.strip():
        return f"System: {system_prompt}\nUser: {question_text}\nAssistant:"
    return f"User: {question_text}\nAssistant:"


def generate_response(model, tokenizer, prompt_text: str, generation_config: dict[str, Any]) -> dict[str, Any]:
    import torch

    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    device = model_input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_token_count = int(encoded["input_ids"].shape[-1])

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_config,
        )

    generated_ids = output_ids[0, prompt_token_count:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "prompt_text": prompt_text,
        "prompt_token_count": prompt_token_count,
        "response_text": response_text.strip(),
        "response_token_count": int(generated_ids.shape[-1]),
    }


def pooled_post_mlp_residuals(model, tokenizer, prompt_text: str, response_text: str) -> np.ndarray:
    import torch

    full_text = f"{prompt_text}{response_text}"
    encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    device = model_input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1]
    response_start = prompt_tokens
    seq_len = int(encoded["input_ids"].shape[-1])
    if response_start >= seq_len:
        raise ValueError("No response tokens available for pooling.")

    pooled: list[np.ndarray | None] = [None] * len(model.model.layers)
    hooks = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, outputs):
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            response_slice = hidden_states[:, response_start:, :]
            pooled_tensor = response_slice.mean(dim=1).detach().to(dtype=torch.float16).cpu().numpy()[0]
            pooled[layer_index] = pooled_tensor
        return hook

    for layer_index, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(layer_index)))

    try:
        with torch.no_grad():
            model(**encoded, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    stacked = [tensor for tensor in pooled if tensor is not None]
    if len(stacked) != len(model.model.layers):
        raise RuntimeError("Failed to capture pooled activations for every decoder layer.")
    return np.stack(stacked, axis=0)


def mean_vector(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={matrix.shape}")
    return matrix.mean(axis=0)


def pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={matrix.shape}")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        raise ValueError("Need at least two samples to run PCA.")
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance = (s ** 2) / max(centered.shape[0] - 1, 1)
    total_variance = explained_variance.sum()
    explained_ratio = explained_variance / total_variance if total_variance > 0 else explained_variance
    scores = centered @ vt.T
    return vt, scores, explained_ratio


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
