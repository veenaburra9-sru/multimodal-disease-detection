"""
Evaluation Metrics for Multimodal Disease Detection
Includes: AUC-ROC, Accuracy, Precision, Recall, F1, Brier Score, Calibration
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from typing import Dict, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    average: str = "binary"
) -> Dict[str, float]:
    """
    Compute all evaluation metrics reported in the paper.

    Args:
        y_true (np.ndarray): Ground truth binary labels (N,)
        y_prob (np.ndarray): Predicted probabilities (N,) or (N, num_classes)
        threshold (float): Decision threshold for binary classification
        average (str): 'binary', 'macro', 'micro', 'weighted' for multi-class

    Returns:
        dict of metric_name -> float value
    """
    # Binary predictions
    if y_prob.ndim == 2:
        y_pred = np.argmax(y_prob, axis=1)
        y_score = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
    else:
        y_pred = (y_prob >= threshold).astype(int)
        y_score = y_prob

    metrics = {}

    # AUC-ROC
    try:
        if y_prob.ndim == 2 and y_prob.shape[1] > 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        else:
            metrics["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["auc"] = float("nan")

    # Classification metrics
    metrics["accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"]    = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"]        = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Average Precision (area under PR curve — better for imbalanced data)
    try:
        metrics["avg_precision"] = average_precision_score(y_true, y_score)
    except Exception:
        metrics["avg_precision"] = float("nan")

    # Brier Score (calibration)
    try:
        if y_score.ndim == 1:
            metrics["brier_score"] = brier_score_loss(y_true, y_score)
    except Exception:
        metrics["brier_score"] = float("nan")

    # Specificity (True Negative Rate)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["specificity"] = tn / (tn + fp + 1e-8)
    except Exception:
        metrics["specificity"] = float("nan")

    return metrics


class MetricTracker:
    """
    Accumulates predictions and labels across batches for epoch-level evaluation.

    Usage:
        tracker = MetricTracker()
        for batch in dataloader:
            ...
            tracker.update(targets, probabilities)
        metrics = tracker.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._all_targets = []
        self._all_probs = []

    def update(self, targets: torch.Tensor, probabilities: torch.Tensor):
        self._all_targets.append(targets.detach().cpu().numpy())
        self._all_probs.append(probabilities.detach().cpu().numpy())

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        y_true = np.concatenate(self._all_targets, axis=0)
        y_prob = np.concatenate(self._all_probs, axis=0)
        if y_prob.ndim == 2 and y_prob.shape[1] == 1:
            y_prob = y_prob.squeeze(1)
        return compute_metrics(y_true, y_prob, threshold=threshold)

    def compute_and_reset(self, threshold: float = 0.5) -> Dict[str, float]:
        metrics = self.compute(threshold)
        self.reset()
        return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty-print metrics table."""
    print(f"\n{'='*50}")
    if prefix:
        print(f"  {prefix}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float) and not np.isnan(v):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.clip(y_true + np.random.normal(0, 0.3, 100), 0, 1)

    metrics = compute_metrics(y_true, y_prob)
    print_metrics(metrics, prefix="Test Metrics")

    # Test MetricTracker
    tracker = MetricTracker()
    tracker.update(torch.tensor(y_true[:50]), torch.tensor(y_prob[:50]))
    tracker.update(torch.tensor(y_true[50:]), torch.tensor(y_prob[50:]))
    tracked = tracker.compute()
    print_metrics(tracked, prefix="Tracked Metrics")
