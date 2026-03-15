"""
SHAP-based Feature Attribution and Attention Visualization
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"

Provides:
  - SHAP values for EHR/structured features (Table 9 in paper)
  - Attention weight visualization across modalities
  - GradCAM for imaging (saliency maps)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional, Dict
import shap


# ─────────────────────────────────────────────────────────────
# 1. SHAP ANALYSIS — EHR FEATURES
# ─────────────────────────────────────────────────────────────

class SHAPAnalyzer:
    """
    Computes SHAP values for structured EHR features using KernelExplainer.

    Produces feature importance values like those in Table 9 of the paper:
        - Troponin levels: 0.48
        - Imaging lesion intensity: 0.44
        - HR variability: 0.39
        - Age: 0.31
        - CRP level: 0.27
    """

    def __init__(self, model: nn.Module, ehr_feature_names: List[str], device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.feature_names = ehr_feature_names
        self.model.eval()

    def _ehr_predict_fn(self, ehr_array: np.ndarray) -> np.ndarray:
        """Wrapper for SHAP: predicts from EHR only."""
        ehr_tensor = torch.tensor(ehr_array, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(ehr=ehr_tensor)
        probs = output["probabilities"].cpu().numpy()
        return probs.squeeze(-1) if probs.ndim > 1 else probs

    def compute_shap(
        self,
        background_data: np.ndarray,
        test_data: np.ndarray,
        n_background: int = 50
    ) -> np.ndarray:
        """
        Compute SHAP values using KernelExplainer.

        Args:
            background_data: (N_bg, D) background samples for SHAP baseline
            test_data: (N_test, D) samples to explain
            n_background: Number of background samples to use

        Returns:
            shap_values: (N_test, D) SHAP values
        """
        # Sub-sample background for speed
        idx = np.random.choice(len(background_data), min(n_background, len(background_data)), replace=False)
        background = background_data[idx]

        explainer = shap.KernelExplainer(self._ehr_predict_fn, background)
        shap_values = explainer.shap_values(test_data, nsamples=100)
        return shap_values

    def plot_summary(
        self,
        shap_values: np.ndarray,
        test_data: np.ndarray,
        max_display: int = 10,
        save_path: Optional[str] = None
    ):
        """Generate SHAP summary (beeswarm) plot."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            test_data,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("SHAP Feature Attribution — EHR Modality", fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def get_top_features(self, shap_values: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Return top-k features by mean absolute SHAP value."""
        mean_abs = np.abs(shap_values).mean(axis=0)
        ranked_idx = np.argsort(mean_abs)[::-1][:top_k]
        return [
            {"feature": self.feature_names[i], "shap_contribution": float(mean_abs[i])}
            for i in ranked_idx
        ]


# ─────────────────────────────────────────────────────────────
# 2. ATTENTION WEIGHT VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_attention_weights(
    attention_weights: np.ndarray,
    modality_names: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize cross-modal attention weights across samples.

    Args:
        attention_weights: (N, M) array — N samples, M modalities
        modality_names: Names of modalities
        save_path: Optional path to save figure
    """
    if modality_names is None:
        modality_names = ["Imaging (CNN)", "ECG (LSTM)", "EHR (MLP)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of attention weights
    im = axes[0].imshow(attention_weights[:50].T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_yticks(range(len(modality_names)))
    axes[0].set_yticklabels(modality_names)
    axes[0].set_xlabel("Sample Index")
    axes[0].set_title("Cross-Modal Attention Weights (First 50 Samples)")
    plt.colorbar(im, ax=axes[0])

    # Mean weights across samples (bar chart)
    mean_weights = attention_weights.mean(axis=0)
    colors = ["#2196F3", "#F44336", "#4CAF50"]
    bars = axes[1].bar(modality_names, mean_weights, color=colors, edgecolor="black")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean Attention Weight (α)")
    axes[1].set_title("Average Modality Contribution")
    for bar, val in zip(bars, mean_weights):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Cross-Modal Attention Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 3. GRADCAM FOR IMAGING
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Maps for CNN branch.
    Highlights regions of X-ray that most influenced the prediction.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            image: (1, 3, H, W) input image
            class_idx: Target class index

        Returns:
            cam: (H, W) heatmap normalized to [0, 1]
        """
        self.model.eval()
        output = self.model(images=image)
        logits = output["logits"]

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(self, image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay GradCAM heatmap on original image."""
        import cv2
        cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
        heatmap = cm.jet(cam_resized)[:, :, :3]
        overlay = alpha * heatmap + (1 - alpha) * image_np / 255.0
        return np.clip(overlay, 0, 1)


# ─────────────────────────────────────────────────────────────
# 4. RESULTS PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training/validation loss and AUC curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_auc"], "b-o", label="Train AUC", markersize=3)
    ax2.plot(epochs, history["val_auc"], "r-o", label="Val AUC", markersize=3)
    ax2.axhline(y=0.94, color="green", linestyle="--", label="Paper AUC (0.94)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title("AUC-ROC Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_performance_comparison(save_path: Optional[str] = None):
    """Reproduce Figure 4 from the paper: unimodal vs multimodal performance."""
    configs = ["Imaging\nonly", "EHR\nonly", "ECG\nonly", "Imaging\n+EHR", "Tri-modal\n(Ours)"]
    auc    = [0.85, 0.83, 0.80, 0.92, 0.94]
    acc    = [82.1, 80.3, 78.6, 87.5, 89.8]
    f1     = [0.81, 0.78, 0.76, 0.87, 0.89]
    recall = [0.78, 0.76, 0.72, 0.85, 0.88]

    x = np.arange(len(configs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*width, auc,           width, label="AUC",      color="#2196F3")
    ax.bar(x - 0.5*width, [a/100 for a in acc], width, label="Accuracy", color="#4CAF50")
    ax.bar(x + 0.5*width, f1,            width, label="F1-Score", color="#FF9800")
    ax.bar(x + 1.5*width, recall,        width, label="Recall",   color="#F44336")

    ax.set_xlabel("Model Configuration", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Comparison: Unimodal vs Multimodal Configurations", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Test attention visualization with random data
    np.random.seed(42)
    fake_weights = np.random.dirichlet([1, 1, 1], size=100)  # (100, 3) attention weights
    plot_attention_weights(fake_weights, save_path="results/attention_weights.png")

    # Test performance comparison plot
    plot_performance_comparison(save_path="results/performance_comparison.png")
