from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import DataCollatorWithPadding


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _truncate_split(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    max_samples = min(max_samples, len(dataset))
    return dataset.select(range(max_samples))


def load_raw_datasets(config: dict[str, Any], project_root: Path) -> DatasetDict:
    dataset_cfg = config["dataset"]
    source = dataset_cfg["source"]

    if source == "ag_news":
        dataset = load_dataset(dataset_cfg.get("name", "ag_news"))
        return DatasetDict(
            train=_truncate_split(dataset["train"], dataset_cfg.get("max_train_samples")),
            validation=_truncate_split(dataset["test"], dataset_cfg.get("max_eval_samples")),
            test=_truncate_split(dataset["test"], dataset_cfg.get("max_eval_samples")),
        )

    if source == "local_debug":
        data_dir = (project_root / dataset_cfg["data_dir"]).resolve()
        return DatasetDict(
            train=Dataset.from_list(_truncate_rows(_read_jsonl(data_dir / "train.jsonl"), dataset_cfg.get("max_train_samples"))),
            validation=Dataset.from_list(_truncate_rows(_read_jsonl(data_dir / "validation.jsonl"), dataset_cfg.get("max_eval_samples"))),
            test=Dataset.from_list(_truncate_rows(_read_jsonl(data_dir / "test.jsonl"), dataset_cfg.get("max_eval_samples"))),
        )

    raise ValueError(f"Unsupported dataset source: {source}")


def _truncate_rows(rows: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    if max_samples is None:
        return rows
    return rows[:max_samples]


def tokenize_datasets(
    raw_datasets: DatasetDict,
    tokenizer,
    config: dict[str, Any],
) -> DatasetDict:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    text_column = dataset_cfg["text_column"]
    label_column = dataset_cfg["label_column"]
    max_length = model_cfg["max_length"]

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = batch[label_column]
        return encoded

    tokenized = raw_datasets.map(
        tokenize_batch,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    return tokenized


def build_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)
