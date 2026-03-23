# Local Offline Result Snapshot

## Purpose

This document records the lightweight local experiments that were run on the current machine.

These runs are intended to verify the implementation end to end with minimal compute. They do not replace the final CUDA experiments because the local environment has no cached pretrained text model available offline.

## Runs Executed

### Full finetuning

- Run directory: `outputs/local-results-full-ft-20260323-172255`
- Config: `configs/local_results_full_ft.yaml`
- Trainable parameters: 227,428
- Total parameters: 227,428
- Validation accuracy: 0.50
- Validation macro F1: 0.3333
- Test accuracy: 0.50
- Test macro F1: 0.4167

### OFT

- Run directory: `outputs/local-results-oft-20260323-172255`
- Config: `configs/local_results_oft.yaml`
- Trainable parameters: 15,460
- Total parameters: 242,888
- Validation accuracy: 0.25
- Validation macro F1: 0.10
- Test accuracy: 0.25
- Test macro F1: 0.10

## Takeaways

- The code path for tokenization, training, evaluation, artifact generation, model saving, and model reload works locally.
- OFT is strongly constrained in this offline setup because the backbone is not pretrained.
- The final comparison should be performed with a pretrained model and the AG News dataset in the later CUDA environment.
