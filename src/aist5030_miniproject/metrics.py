from __future__ import annotations

from typing import Iterable


def accuracy_score(predictions: Iterable[int], references: Iterable[int]) -> float:
    preds = list(predictions)
    refs = list(references)
    if not refs:
        return 0.0
    correct = sum(int(pred == ref) for pred, ref in zip(preds, refs))
    return correct / len(refs)


def macro_f1_score(predictions: Iterable[int], references: Iterable[int], num_labels: int) -> float:
    preds = list(predictions)
    refs = list(references)
    if not refs:
        return 0.0

    f1_values = []
    for label in range(num_labels):
        tp = sum(1 for pred, ref in zip(preds, refs) if pred == label and ref == label)
        fp = sum(1 for pred, ref in zip(preds, refs) if pred == label and ref != label)
        fn = sum(1 for pred, ref in zip(preds, refs) if pred != label and ref == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append((2 * precision * recall) / (precision + recall))

    return sum(f1_values) / num_labels

