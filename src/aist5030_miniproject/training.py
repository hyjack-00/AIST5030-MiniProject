from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import clone_config, load_config, save_yaml
from .data import build_data_collator, load_raw_datasets, tokenize_datasets
from .evaluation import evaluate_split
from .io_utils import save_markdown, save_predictions_csv
from .modeling import (
    apply_finetuning_strategy,
    align_model_and_tokenizer,
    build_base_model,
    count_parameters,
    filter_batch_for_model,
    load_tokenizer,
    move_batch_to_device,
)
from .plots import plot_loss_curve
from .utils import append_jsonl, choose_device, ensure_dir, format_parameter_count, save_json, set_seed, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sequence classification model with full finetuning or OFT.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def build_run_directory(config: dict[str, Any], project_root: Path) -> Path:
    output_root = ensure_dir(project_root / config.get("output_root", "outputs"))
    experiment_name = config["experiment_name"]
    run_dir = output_root / f"{experiment_name}-{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float,
    log_every_steps: int,
    starting_global_step: int,
) -> list[dict[str, float]]:
    model.train()
    log_records: list[dict[str, float]] = []
    optimizer.zero_grad()
    global_step = starting_global_step
    last_outputs = None

    for step_index, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        batch = filter_batch_for_model(model, batch)
        outputs = model(**batch)
        last_outputs = outputs
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if step_index % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % log_every_steps == 0 or global_step == 1:
                log_records.append(
                    {
                        "global_step": global_step,
                        "loss": float(outputs.loss.detach().cpu()),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                    }
                )

    if last_outputs is not None and len(dataloader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        if global_step % log_every_steps == 0 or global_step == 1:
            log_records.append(
                {
                    "global_step": global_step,
                    "loss": float(last_outputs.loss.detach().cpu()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )

    return log_records


def save_run_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    model,
    tokenizer,
    train_log: list[dict[str, float]],
    metrics: dict[str, float],
    prediction_rows: list[dict[str, str]],
    parameter_stats: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    save_yaml(config, run_dir / "config.snapshot.yaml")
    save_json(metrics, run_dir / "metrics.json")
    save_json(parameter_stats, run_dir / "parameter_stats.json")
    append_jsonl(train_log, run_dir / "train_log.jsonl")
    save_predictions_csv(prediction_rows, run_dir / "sample_predictions.csv")
    plot_loss_curve(train_log, run_dir / "loss_curve.png")
    save_markdown(build_summary_markdown(config, metrics, parameter_stats, metadata), run_dir / "run_summary.md")

    model_dir = ensure_dir(run_dir / "model")
    tokenizer.save_pretrained(model_dir)
    torch.save(model.state_dict(), model_dir / "state_dict.pt")
    save_json(metadata, run_dir / "run_state.json")


def build_summary_markdown(
    config: dict[str, Any],
    metrics: dict[str, float],
    parameter_stats: dict[str, int],
    metadata: dict[str, Any],
) -> str:
    lines = [
        f"# Run Summary: {config['experiment_name']}",
        "",
        "## Setup",
        "",
        f"- Finetuning mode: {config['training']['finetuning_mode']}",
        f"- Dataset source: {config['dataset']['source']}",
        f"- Architecture: {config['model']['architecture']}",
        f"- Use pretrained weights: {config['model']['use_pretrained']}",
        f"- Trainable parameters: {format_parameter_count(parameter_stats['trainable_parameters'])}",
        f"- Total parameters: {format_parameter_count(parameter_stats['total_parameters'])}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in sorted(metrics.items()):
        lines.append(f"- {key}: {value:.4f}")

    lines.extend(
        [
            "",
            "## Metadata",
            "",
            f"- Resolved OFT target modules: {metadata.get('resolved_target_modules', [])}",
            f"- Device used: {metadata.get('device')}",
            "",
        ]
    )
    return "\n".join(lines)


def run_training(config_path: str | Path) -> Path:
    project_root = Path(config_path).resolve().parent.parent
    config = load_config(config_path)
    frozen_config = clone_config(config)
    set_seed(config["seed"])
    device = choose_device(config.get("device", "auto"))

    run_dir = build_run_directory(config, project_root)

    tokenizer = load_tokenizer(config, project_root)
    raw_datasets = load_raw_datasets(config, project_root)
    tokenized = tokenize_datasets(raw_datasets, tokenizer, config)
    collator = build_data_collator(tokenizer)

    model = build_base_model(config, project_root)
    align_model_and_tokenizer(model, tokenizer)
    model, extra_metadata = apply_finetuning_strategy(model, config)
    model.to(device)

    parameter_stats = count_parameters(model)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=config["training"].get("num_workers", 0),
    )

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    gradient_accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    total_update_steps = math.ceil(len(train_loader) / gradient_accumulation_steps) * int(config["training"]["epochs"])
    warmup_steps = int(total_update_steps * float(config["training"].get("warmup_ratio", 0.0)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_update_steps, 1),
    )

    train_log: list[dict[str, float]] = []
    global_step = 0
    for _epoch in range(int(config["training"]["epochs"])):
        epoch_log = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clip_norm=float(config["training"].get("gradient_clip_norm", 1.0)),
            log_every_steps=int(config["training"].get("log_every_steps", 1)),
            starting_global_step=global_step,
        )
        train_log.extend(epoch_log)
        if epoch_log:
            global_step = int(epoch_log[-1]["global_step"])

    metrics: dict[str, float] = {}
    prediction_rows: list[dict[str, str]] = []
    for split_name in ["validation", "test"]:
        dataloader = DataLoader(
            tokenized[split_name],
            batch_size=config["training"]["eval_batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=config["training"].get("num_workers", 0),
        )
        split_metrics, split_predictions = evaluate_split(
            model=model,
            split_name=split_name,
            dataset=tokenized[split_name],
            raw_split=raw_datasets[split_name],
            dataloader=dataloader,
            device=device,
            label_names=config["dataset"]["label_names"],
            max_prediction_rows=config["dataset"].get("max_predict_samples", 16),
        )
        metrics.update(split_metrics)
        prediction_rows.extend(split_predictions)

    metadata = {
        "device": str(device),
        "resolved_target_modules": extra_metadata.get("resolved_target_modules", []),
        "finetuning_mode": config["training"]["finetuning_mode"],
    }
    save_run_artifacts(
        run_dir=run_dir,
        config=frozen_config,
        model=model,
        tokenizer=tokenizer,
        train_log=train_log,
        metrics=metrics,
        prediction_rows=prediction_rows,
        parameter_stats=parameter_stats,
        metadata=metadata,
    )
    return run_dir


def main() -> None:
    args = parse_args()
    run_dir = run_training(args.config)
    print(f"Saved training artifacts to {run_dir}")
