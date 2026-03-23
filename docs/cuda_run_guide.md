# CUDA Run Guide

## Scope

Use this guide when running the final experiments on a CUDA machine.

This project is currently single-process only. Even on an 8-GPU node, the simplest and safest way is to run one job on one selected GPU.

## 1. Setup

Clone the repo and enter it:

```bash
git clone <your-repo-url>
cd AIST5030-MiniProject
```

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Check CUDA:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 2. Recommended Run Order

Run the baseline first, then OFT.

Use one GPU explicitly:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Baseline:

```bash
python train.py --config configs/ag_news_full_ft.yaml
```

OFT:

```bash
python train.py --config configs/ag_news_oft.yaml
```

If you want a larger backbone, use the BERT-base configs instead:

```bash
python train.py --config configs/ag_news_bert_base_full_ft.yaml
python train.py --config configs/ag_news_bert_base_oft.yaml
```

If you want to use a local Qwen model, place the model directory here first:

```bash
local_models/Qwen2.5-0.5B-Instruct
```

Then run:

```bash
python train.py --config configs/ag_news_qwen25_05b_full_ft_local.yaml
python train.py --config configs/ag_news_qwen25_05b_oft_local.yaml
```

## 3. Evaluate Saved Runs

Each training run writes a new folder under `outputs/`.

Example:

```bash
python evaluate.py --checkpoint outputs/ag-news-full-ft-<timestamp>
python evaluate.py --checkpoint outputs/ag-news-oft-<timestamp>
```

## 4. Files to Collect for the Report

For each run, keep these files:

- `metrics.json`
- `parameter_stats.json`
- `loss_curve.png`
- `sample_predictions.csv`
- `run_summary.md`

## 5. Expected Result Quality

The final comparison should be based on:

- `configs/ag_news_full_ft.yaml`
- `configs/ag_news_oft.yaml`
- a pretrained backbone downloaded from Hugging Face
- the real AG News dataset

Do not use the offline local debug runs as final report results.

## 6. If You Want to Use a Different GPU

Change only this line:

```bash
export CUDA_VISIBLE_DEVICES=3
```

## 7. If You Want to Try Larger Runs Later

The simplest knobs to change are:

- `training.epochs`
- `training.batch_size`
- `training.eval_batch_size`
- `training.learning_rate`
- `dataset.max_eval_samples`

Edit the YAML config directly and keep the original files unchanged by copying them first.

## 8. Recommended Bigger Model

If you want a larger model than DistilBERT without making the run too risky, use:

- `bert-base-uncased`

Why this one:

- clearly larger than `distilbert-base-uncased`
- already supported by the current OFT target-module mapping
- safer than jumping straight to much larger models before the first successful run

If you specifically want Qwen, the project now includes local-model configs for `Qwen2.5-0.5B-Instruct`.
Those configs assume the downloaded model is already present on disk and do not access the Hugging Face Hub at runtime.
