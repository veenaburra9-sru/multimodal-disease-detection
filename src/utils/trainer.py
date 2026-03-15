"""
Training Loop for Multimodal Disease Detection Model
Implements Adam optimizer with learning rate scheduling and early stopping.
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"

Optimization (Eq. 11): θ* = argmin_θ L(θ)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional
import numpy as np

from .losses import MultimodalLoss
from .metrics import MetricTracker, print_metrics


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class Trainer:
    """
    End-to-end trainer for the multimodal model.

    Args:
        model: MultimodalDiseaseDetector
        loss_fn: MultimodalLoss
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        lr (float): Initial learning rate (default 0.001 from paper)
        weight_decay (float): L2 regularization via Adam
        patience (int): Early stopping patience
        save_dir (str): Directory to save checkpoints
        device (str): 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: MultimodalLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        save_dir: str = "results/checkpoints",
        device: str = "cuda"
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # Optimizer: Adam (Eq. 11 — gradient-based method)
        self.optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # LR scheduler: reduce on validation loss plateau
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        self.early_stopping = EarlyStopping(patience=patience, mode="min")
        self.train_metric_tracker = MetricTracker()
        self.val_metric_tracker = MetricTracker()

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_auc": [], "val_auc": []
        }

        self.model.to(self.device)
        print(f"Training on: {self.device}")

    def _batch_to_device(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _forward_batch(self, batch: Dict):
        """Run model forward pass on a batch."""
        images = batch.get("image", None)
        ecg    = batch.get("ecg", None)
        ehr    = batch.get("ehr", None)
        labels = batch["label"]

        output = self.model(images=images, ecg=ecg, ehr=ehr)
        return output, labels

    def train_epoch(self) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        self.train_metric_tracker.reset()

        for batch in self.train_loader:
            batch = self._batch_to_device(batch)
            self.optimizer.zero_grad()

            output, labels = self._forward_batch(batch)

            losses = self.loss_fn(
                logits=output["logits"],
                targets=labels,
                modality_features=None  # Skip alignment loss in batch for speed
            )

            loss = losses["total"]
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

            self.train_metric_tracker.update(labels, output["probabilities"])

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_epoch(self) -> float:
        """Run validation. Returns mean val loss."""
        self.model.eval()
        total_loss = 0.0
        self.val_metric_tracker.reset()

        for batch in self.val_loader:
            batch = self._batch_to_device(batch)
            output, labels = self._forward_batch(batch)

            losses = self.loss_fn(logits=output["logits"], targets=labels)
            total_loss += losses["total"].item()

            self.val_metric_tracker.update(labels, output["probabilities"])

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history
        }
        path = os.path.join(self.save_dir, f"epoch_{epoch:03d}.pth")
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    def fit(self, num_epochs: int = 100, log_interval: int = 1) -> Dict:
        """
        Full training loop.

        Args:
            num_epochs: Maximum number of epochs
            log_interval: Print metrics every N epochs

        Returns:
            Training history dict
        """
        best_val_loss = float("inf")
        print(f"\nStarting training for up to {num_epochs} epochs...\n")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            # Train
            train_loss = self.train_epoch()
            train_metrics = self.train_metric_tracker.compute()

            # Validate
            val_loss = self.validate_epoch()
            val_metrics = self.val_metric_tracker.compute()

            # LR scheduling
            self.scheduler.step(val_loss)

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_auc"].append(train_metrics.get("auc", 0))
            self.history["val_auc"].append(val_metrics.get("auc", 0))

            elapsed = time.time() - t0

            if epoch % log_interval == 0:
                print(
                    f"Epoch [{epoch:>3}/{num_epochs}] | "
                    f"Train Loss: {train_loss:.4f} | AUC: {train_metrics.get('auc', 0):.4f} | "
                    f"Val Loss: {val_loss:.4f} | AUC: {val_metrics.get('auc', 0):.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Save best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss, is_best=is_best)

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        print("\nTraining complete.")
        print_metrics(val_metrics, prefix="Final Validation Metrics")
        return self.history
