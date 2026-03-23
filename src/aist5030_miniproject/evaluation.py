from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import load_config, save_yaml
from .data import build_data_collator, load_raw_datasets, tokenize_datasets
from .io_utils import save_markdown, save_predictions_csv
from .metrics import accuracy_score, macro_f1_score
from .modeling import apply_finetuning_strategy, build_base_model, filter_batch_for_model, load_tokenizer, move_batch_to_device
from .utils import choose_device, save_json


def evaluate_split(
    model,
    split_name: str,
    dataset,
    raw_split,
    dataloader: DataLoader,
    device: torch.device,
    label_names: list[str],
    max_prediction_rows: int,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    model.eval()
    losses = []
    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            batch = filter_batch_for_model(model, batch)
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            refs = batch["labels"].detach().cpu().tolist()
            predictions.extend(preds)
            references.extend(refs)

    metrics = {
        f"{split_name}_loss": sum(losses) / max(len(losses), 1),
        f"{split_name}_accuracy": accuracy_score(predictions, references),
        f"{split_name}_macro_f1": macro_f1_score(predictions, references, len(label_names)),
    }

    prediction_rows = []
    for index, (prediction, reference) in enumerate(zip(predictions, references)):
        if index >= max_prediction_rows:
            break
        prediction_rows.append(
            {
                "text": raw_split[index]["text"],
                "label": label_names[reference],
                "prediction": label_names[prediction],
                "split": split_name,
            }
        )
    return metrics, prediction_rows


def load_model_from_run(checkpoint_dir: Path):
    run_state = load_json(checkpoint_dir / "run_state.json")
    config = load_config(checkpoint_dir / "config.snapshot.yaml")
    project_root = checkpoint_dir.parent.parent
    tokenizer = load_tokenizer(config, project_root)

    model = build_base_model(config, project_root)
    model, _ = apply_finetuning_strategy(model, config)

    state_dict = torch.load(checkpoint_dir / "model" / "state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model, tokenizer, config, run_state


def load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_evaluation(checkpoint_dir: str | Path, config_override: str | None = None) -> Path:
    checkpoint_path = Path(checkpoint_dir).resolve()
    config = load_config(config_override) if config_override else load_config(checkpoint_path / "config.snapshot.yaml")
    project_root = checkpoint_path.parent.parent
    device = choose_device(config.get("device", "auto"))
    label_names = config["dataset"]["label_names"]

    tokenizer = load_tokenizer(config, project_root)
    raw_datasets = load_raw_datasets(config, project_root)
    tokenized = tokenize_datasets(raw_datasets, tokenizer, config)
    collator = build_data_collator(tokenizer)

    model = build_base_model(config, project_root)
    model, _ = apply_finetuning_strategy(model, config)
    state_dict = torch.load(checkpoint_path / "model" / "state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    metrics: dict[str, float] = {}
    prediction_rows: list[dict[str, str]] = []

    for split_name in ["validation", "test"]:
        dataloader = DataLoader(
            tokenized[split_name],
            batch_size=config["training"]["eval_batch_size"],
            shuffle=False,
            collate_fn=collator,
        )
        split_metrics, split_predictions = evaluate_split(
            model=model,
            split_name=split_name,
            dataset=tokenized[split_name],
            raw_split=raw_datasets[split_name],
            dataloader=dataloader,
            device=device,
            label_names=label_names,
            max_prediction_rows=config["dataset"].get("max_predict_samples", 16),
        )
        metrics.update(split_metrics)
        prediction_rows.extend(split_predictions)

    save_json(metrics, checkpoint_path / "evaluation.metrics.json")
    save_predictions_csv(prediction_rows, checkpoint_path / "evaluation.sample_predictions.csv")
    save_markdown(_format_eval_summary(metrics), checkpoint_path / "evaluation.summary.md")
    save_yaml(config, checkpoint_path / "evaluation.config.snapshot.yaml")
    return checkpoint_path


def _format_eval_summary(metrics: dict[str, float]) -> str:
    lines = ["# Evaluation Summary", ""]
    for key, value in sorted(metrics.items()):
        lines.append(f"- {key}: {value:.4f}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved OFT or full finetuning run.")
    parser.add_argument("--checkpoint", required=True, help="Path to a run directory under outputs/.")
    parser.add_argument("--config", default=None, help="Optional config override path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_evaluation(args.checkpoint, args.config)
    print(f"Saved evaluation artifacts to {run_dir}")
