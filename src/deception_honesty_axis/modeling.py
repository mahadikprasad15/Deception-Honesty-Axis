from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def load_model_and_tokenizer(model_config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if model_config.get("device", "auto") == "auto" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=resolve_dtype(model_config.get("torch_dtype", "float16")),
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model


def build_messages(system_prompt: str, user_question: str, assistant_response: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})
    return messages


def generate_responses(
    tokenizer,
    model,
    batch_messages: list[list[dict[str, str]]],
    generation_config: dict[str, Any],
) -> list[str]:
    rendered = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in batch_messages
    ]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_config["max_new_tokens"],
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            do_sample=generation_config["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded: list[str] = []
    for row, input_length in zip(outputs, input_lengths):
        decoded.append(tokenizer.decode(row[int(input_length):], skip_special_tokens=True).strip())
    return decoded
