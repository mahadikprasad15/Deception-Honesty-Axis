from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import torch

from deception_honesty_axis.sycophancy_activations import normalize_activation_pooling, pool_hidden_states


class ActivationTextRecord(Protocol):
    prompt_text: str
    activation_text: str


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_transformer_layers(model) -> Any:  # noqa: ANN001
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers at model.model.layers or model.transformer.h")


def load_tokenizer_and_model(model_name: str, dtype: torch.dtype, device: str):  # noqa: ANN201
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def extract_batch_activations(
    *,
    tokenizer,
    model,
    records: list[ActivationTextRecord],
    layer_number: int,
    pooling: str,
    add_special_tokens: bool,
    max_length: int | None,
) -> list[dict[str, Any]]:
    layers = resolve_transformer_layers(model)
    if layer_number < 1 or layer_number > len(layers):
        raise ValueError(f"Layer {layer_number} is out of range for model with {len(layers)} layers")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output) -> None:  # noqa: ANN001
        captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = layers[layer_number - 1].register_forward_hook(hook)
    try:
        encoded = tokenizer(
            [record.activation_text for record in records],
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        prompt_encoded = tokenizer(
            [record.prompt_text for record in records],
            return_tensors="pt",
            padding=True,
            add_special_tokens=add_special_tokens,
        )
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        if "attention_mask" not in prompt_encoded:
            prompt_encoded["attention_mask"] = torch.ones_like(prompt_encoded["input_ids"])
        prompt_token_counts = [int(value) for value in prompt_encoded["attention_mask"].sum(dim=1).tolist()]
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()

    pooled, pooling_metadata = pool_hidden_states(
        captured["hidden"],
        encoded["attention_mask"],
        prompt_token_counts,
        pooling,
    )
    activations = pooled.to(dtype=torch.float32, device="cpu")
    return [
        {
            "activation": [float(value) for value in activation.tolist()],
            "prompt_token_count": int(metadata.prompt_token_count),
            "response_token_count": int(metadata.response_token_count),
            "used_last_token_fallback": bool(metadata.used_last_token_fallback),
        }
        for activation, metadata in zip(activations, pooling_metadata, strict=True)
    ]


def _normalize_record_field_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return int(default)
    return int(value)


def normalize_saved_records(rows: list[dict[str, Any]], *, layer_number: int, pooling: str) -> list[dict[str, Any]]:
    normalized_pooling = normalize_activation_pooling(pooling) or "mean_response"
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("ids", ""))
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        payload = dict(row)
        payload["activation_pooling"] = str(
            normalize_activation_pooling(payload.get("activation_pooling")) or normalized_pooling
        )
        payload["activation_layer_number"] = _normalize_record_field_int(
            payload.get("activation_layer_number"),
            layer_number,
        )
        payload["prompt_token_count"] = _normalize_record_field_int(payload.get("prompt_token_count"), 0)
        payload["response_token_count"] = _normalize_record_field_int(payload.get("response_token_count"), 0)
        payload["used_last_token_fallback"] = bool(payload.get("used_last_token_fallback", False))
        normalized.append(payload)
    return normalized


def save_activation_rows_dataset(
    records: list[dict[str, Any]],
    *,
    run_root: Path,
    push_repo_id: str | None,
    private: bool,
) -> None:
    from datasets import Dataset

    dataset = Dataset.from_list(records)
    dataset_dir = run_root / "results" / "hf_dataset"
    dataset.save_to_disk(str(dataset_dir))
    if push_repo_id:
        dataset.push_to_hub(push_repo_id, private=private, token=os.environ.get("HF_TOKEN"))
