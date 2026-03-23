from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from peft import OFTConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizerFast, DistilBertConfig

DEFAULT_OFT_TARGETS = {
    "distilbert": ["q_lin", "k_lin", "v_lin", "out_lin"],
    "bert": ["query", "key", "value", "dense"],
    "roberta": ["query", "key", "value", "dense"],
}


def build_label_mappings(config: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    label_names = config["dataset"]["label_names"]
    id2label = {idx: label for idx, label in enumerate(label_names)}
    label2id = {label: idx for idx, label in enumerate(label_names)}
    return id2label, label2id


def load_tokenizer(config: dict[str, Any], project_root: Path):
    model_cfg = config["model"]
    tokenizer_name = model_cfg.get("tokenizer_name_or_path") or model_cfg.get("model_name_or_path")
    if not tokenizer_name:
        raise ValueError("A tokenizer path or model name is required.")

    if model_cfg["use_pretrained"]:
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=model_cfg.get("local_files_only", False),
            use_fast=True,
        )

    tokenizer_path = (project_root / tokenizer_name).resolve()
    return BertTokenizerFast.from_pretrained(str(tokenizer_path), local_files_only=True, do_lower_case=True)


def build_base_model(config: dict[str, Any], project_root: Path):
    model_cfg = config["model"]
    id2label, label2id = build_label_mappings(config)
    num_labels = len(id2label)

    if model_cfg["use_pretrained"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["model_name_or_path"],
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            local_files_only=model_cfg.get("local_files_only", False),
        )
        return model

    architecture = model_cfg["architecture"]
    if architecture != "distilbert":
        raise ValueError(f"Offline config mode is only implemented for distilbert, received {architecture}.")

    offline_cfg = model_cfg["offline_config"]
    tokenizer_path = (project_root / model_cfg["tokenizer_name_or_path"]).resolve()
    vocab_size = offline_cfg["vocab_size"]
    vocab_file = tokenizer_path / "vocab.txt"
    if vocab_file.exists():
        with vocab_file.open("r", encoding="utf-8") as handle:
            vocab_size = max(vocab_size, sum(1 for _ in handle))

    hf_config = DistilBertConfig(
        vocab_size=vocab_size,
        dim=offline_cfg["dim"],
        hidden_dim=offline_cfg["hidden_dim"],
        n_layers=offline_cfg["n_layers"],
        n_heads=offline_cfg["n_heads"],
        dropout=offline_cfg.get("dropout", 0.1),
        attention_dropout=offline_cfg.get("attention_dropout", 0.1),
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return AutoModelForSequenceClassification.from_config(hf_config)


def resolve_oft_target_modules(config: dict[str, Any]) -> list[str]:
    requested = config["oft"].get("target_modules")
    if requested:
        return requested
    architecture = config["model"]["architecture"]
    if architecture not in DEFAULT_OFT_TARGETS:
        raise ValueError(
            f"No default OFT target modules defined for architecture '{architecture}'. "
            "Set oft.target_modules explicitly in the config."
        )
    return DEFAULT_OFT_TARGETS[architecture]


def apply_finetuning_strategy(model, config: dict[str, Any]):
    mode = config["training"]["finetuning_mode"]
    if mode == "full_ft":
        return model, {"resolved_target_modules": []}
    if mode != "oft":
        raise ValueError(f"Unsupported finetuning mode: {mode}")

    oft_cfg = config["oft"]
    target_modules = resolve_oft_target_modules(config)
    peft_kwargs = {
        "task_type": TaskType.SEQ_CLS,
        "target_modules": target_modules,
        "modules_to_save": oft_cfg.get("modules_to_save"),
        "bias": oft_cfg.get("bias", "none"),
    }
    if oft_cfg.get("r") is not None:
        peft_kwargs["r"] = oft_cfg["r"]
    if oft_cfg.get("oft_block_size") is not None:
        peft_kwargs["oft_block_size"] = oft_cfg["oft_block_size"]
    peft_config = OFTConfig(**peft_kwargs)
    model = get_peft_model(model, peft_config)
    return model, {"resolved_target_modules": target_modules}


def count_parameters(model) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def filter_batch_for_model(model, batch: dict[str, Any]) -> dict[str, Any]:
    accepted = set(inspect.signature(model.forward).parameters.keys())
    return {key: value for key, value in batch.items() if key in accepted}
