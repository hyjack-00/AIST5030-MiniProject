# AIST5030 Mini Project: Orthogonal Finetuning for Text Classification

This repository implements a lightweight, reproducible mini project for parameter-efficient finetuning with Orthogonal Finetuning (OFT).

The project focuses on AG News text classification and supports two modes:

- An offline sanity mode that uses a tiny local debug dataset and a compact DistilBERT-style model instantiated from config.
- A GPU-ready mode that switches to AG News and a pretrained Hugging Face checkpoint once a CUDA environment with model/data access is available.

## Project Goals

- Compare full finetuning against OFT on the same sequence classification task.
- Save report-ready artifacts such as loss curves, metrics, and sample predictions.
- Keep the local validation path cheap enough to run on CPU.

## Repository Layout

- `train.py`: Training entry point.
- `evaluate.py`: Standalone evaluation entry point.
- `configs/`: Sanity and GPU-ready experiment configs.
- `data/local_debug/`: Tiny offline text classification dataset.
- `assets/offline_bert_tokenizer/`: Minimal tokenizer assets for sanity mode.
- `docs/plan_report.md`: Persistent requirement extraction and execution plan.
- `docs/report_outline.md`: Draft structure for the final 3-page report.
- `src/aist5030_miniproject/`: Core project modules.

## Environment

The current machine exposes different `python3` interpreters depending on the working directory. The project was implemented against the Conda interpreter that already includes `torch`, `transformers`, `datasets`, `peft`, `matplotlib`, and `pyyaml`:

```bash
/opt/miniconda3/bin/python3
```

If you prefer a virtual environment, install the dependencies listed in `requirements.txt`.

## Quick Start

Run the offline sanity checks first:

```bash
/opt/miniconda3/bin/python3 train.py --config configs/sanity_full_ft.yaml
/opt/miniconda3/bin/python3 train.py --config configs/sanity_oft.yaml
```

For a slightly more meaningful local result on the tiny offline dataset, run:

```bash
/opt/miniconda3/bin/python3 train.py --config configs/local_results_full_ft.yaml
/opt/miniconda3/bin/python3 train.py --config configs/local_results_oft.yaml
```

Each run writes artifacts to `outputs/<experiment-name>-<timestamp>/`.

To evaluate an existing run:

```bash
/opt/miniconda3/bin/python3 evaluate.py --checkpoint outputs/<run-dir>
```

## GPU-Ready Runs

Once you move to a CUDA machine with model hub access, switch to the AG News configs:

```bash
python train.py --config configs/ag_news_full_ft.yaml
python train.py --config configs/ag_news_oft.yaml
```

If you have already downloaded AG News into `data/ag_news_hf`, use the local-data configs:

```bash
python train.py --config configs/ag_news_local_full_ft.yaml
python train.py --config configs/ag_news_local_oft.yaml
```

These configs assume the following defaults:

- Dataset: `ag_news`
- Backbone: `distilbert-base-uncased`
- Task: 4-class sequence classification

Important:

- The current code uses a single training process.
- On an 8-GPU machine, run on one selected GPU first.
- Use `CUDA_VISIBLE_DEVICES=<gpu_id>` to choose the GPU.

See `docs/cuda_run_guide.md` for the exact CUDA workflow.

If you want a larger ready-made option, use:

- `configs/ag_news_bert_base_full_ft.yaml`
- `configs/ag_news_bert_base_oft.yaml`

If you want to use a local Qwen model, use:

- `configs/ag_news_qwen25_05b_full_ft_local.yaml`
- `configs/ag_news_qwen25_05b_oft_local.yaml`

These configs expect the model files to exist under:

```bash
local_models/Qwen2.5-0.5B-Instruct
```

## Saved Artifacts

Each run produces:

- `metrics.json`: Validation and test metrics.
- `train_log.jsonl`: Step-level training logs.
- `loss_curve.png`: Training loss curve.
- `sample_predictions.csv`: Small qualitative prediction sample.
- `run_summary.md`: Human-readable experiment summary.
- `config.snapshot.yaml`: Frozen config used for the run.
- `model/`: Saved model or adapter files plus tokenizer files.
- `run_state.json`: Metadata required for later reload.

## Notes on Offline Mode

The current environment cannot reliably reach Hugging Face Hub. To keep the project testable now, the sanity configs:

- Use local debug data instead of AG News downloads.
- Instantiate a tiny DistilBERT-style classifier from config instead of downloading a pretrained model.

This keeps the code path stable without blocking the later GPU experiments that satisfy the full assignment setup.
