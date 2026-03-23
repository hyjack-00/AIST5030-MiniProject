# Project Plan Report

## Requirement Extraction

- Assignment theme: parameter-efficient finetuning for pretrained foundation models.
- Required method: Orthogonal Finetuning (OFT).
- Required deliverables:
  - English GitHub repository with readable README.
  - English 3-page report.
- Required evidence in the report:
  - Training loss curves.
  - Final task performance.
  - Before/after finetuning comparison.

## Chosen Project Direction

- Task: 4-class news classification.
- Target dataset for final experiments: AG News.
- Baseline: full finetuning on the same architecture.
- PEFT method: OFT with PEFT.
- Default backbone for final experiments: `distilbert-base-uncased`.

## Execution Strategy

The project is split into two tracks so development can proceed without expensive computation:

1. Offline sanity track
   - Use a tiny local dataset with the same label space as AG News.
   - Use a compact DistilBERT-style model instantiated from config.
   - Verify data processing, model construction, OFT wrapping, logging, plotting, saving, and reload.

2. Final GPU track
   - Switch to the real AG News dataset.
   - Switch to a pretrained Hugging Face checkpoint.
   - Run full finetuning and OFT under the same evaluation protocol.
   - Export report-ready curves and comparison tables.

## Current Deliverables Implemented

- Repository skeleton with training and evaluation entry points.
- Config-driven experiment setup for sanity and final runs.
- Local debug dataset and tokenizer assets for offline execution.
- Artifact generation:
  - metrics JSON
  - training log JSONL
  - loss curve PNG
  - sample predictions CSV
  - run summary Markdown
- Report outline document.

## Local Offline Proxy Results

The current machine does not have an offline cached text backbone from Hugging Face Hub, so the local results below are based on a compact DistilBERT-style model initialized from config rather than pretrained weights.

This is useful for validating the training and evaluation pipeline, but it is not a faithful substitute for the final assignment setting where OFT should adapt a pretrained model.

### Local full finetuning result

- Config: `configs/local_results_full_ft.yaml`
- Trainable parameters: 227,428 / 227,428
- Validation accuracy: 0.50
- Validation macro F1: 0.3333
- Test accuracy: 0.50
- Test macro F1: 0.4167

### Local OFT result

- Config: `configs/local_results_oft.yaml`
- Trainable parameters: 15,460 / 242,888
- Validation accuracy: 0.25
- Validation macro F1: 0.10
- Test accuracy: 0.25
- Test macro F1: 0.10

### Interpretation

- The offline full finetuning baseline can partially fit the tiny debug dataset.
- The offline OFT run uses far fewer trainable parameters, but underperforms because the backbone is not pretrained.
- A meaningful OFT comparison should be rerun in the CUDA environment with a real pretrained encoder and the AG News dataset.

## Planned Final Experiments on CUDA

- Run `configs/ag_news_full_ft.yaml`.
- Run `configs/ag_news_oft.yaml`.
- Record:
  - trainable parameter counts
  - validation/test accuracy
  - macro F1
  - training loss curves
  - example predictions
- Compare efficiency and performance tradeoffs between full finetuning and OFT.

## Risks and Mitigations

- Hub connectivity is unavailable in the current environment.
  - Mitigation: keep a fully offline sanity configuration.
- OFT target modules depend on the encoder architecture.
  - Mitigation: expose `target_modules` in config and keep architecture-aware defaults in code.
- Final report needs concise evidence rather than many experiments.
  - Mitigation: focus on a controlled baseline vs OFT comparison first.
