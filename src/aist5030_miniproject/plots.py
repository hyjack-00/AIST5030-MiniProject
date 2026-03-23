from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_loss_curve(log_records: list[dict], output_path: str | Path) -> None:
    steps = [record["global_step"] for record in log_records]
    losses = [record["loss"] for record in log_records]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, marker="o", linewidth=1.5)
    plt.xlabel("Global step")
    plt.ylabel("Training loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
