"""
Loss Functions for Multimodal Training
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"

Total loss (Eq. 9):
    L = L_classification + λ * L_regularization + μ * L_alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultimodalLoss(nn.Module):
    """
    Combined loss function implementing Eq. 9 from the paper.

    Components:
        L_classification : Binary cross-entropy or multi-class cross-entropy
        L_regularization : L2 weight penalty (model complexity)
        L_alignment      : Contrastive loss to align modal representations in latent space

    Args:
        lambda_reg (float): Weight for regularization loss (λ in paper)
        mu_align (float): Weight for alignment loss (μ in paper)
        num_classes (int): 1 for binary, >1 for multi-class
        class_weights (Tensor, optional): Per-class weights for imbalanced data
        use_focal_loss (bool): Replace BCE with Focal Loss for hard example mining
        focal_gamma (float): Focusing parameter for Focal Loss
    """

    def __init__(
        self,
        lambda_reg: float = 1e-4,
        mu_align: float = 0.1,
        num_classes: int = 1,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        super(MultimodalLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.mu_align = mu_align
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        if num_classes == 1:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Standard cross-entropy or focal loss."""
        if self.num_classes == 1:
            targets = targets.float().unsqueeze(-1) if targets.dim() == 1 else targets.float()
            loss = self.cls_loss_fn(logits, targets)
            if self.use_focal_loss:
                prob = torch.sigmoid(logits)
                pt = torch.where(targets == 1, prob, 1 - prob)
                focal_weight = (1 - pt) ** self.focal_gamma
                loss = (focal_weight * F.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none')).mean()
        else:
            loss = self.cls_loss_fn(logits, targets.long())

        return loss

    def alignment_loss(
        self,
        features: list,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive alignment loss: pulls together same-class features across modalities,
        pushes apart different-class features.

        Args:
            features: List of (B, latent_dim) modality features
            targets: (B,) class labels
        Returns:
            Scalar alignment loss
        """
        if len(features) < 2:
            return torch.tensor(0.0, device=features[0].device)

        total_align = torch.tensor(0.0, device=features[0].device)
        pairs = 0

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                f1 = F.normalize(features[i], dim=-1)
                f2 = F.normalize(features[j], dim=-1)
                sim = (f1 * f2).sum(dim=-1)   # (B,) cosine similarity

                # Same class -> sim near 1; Different class -> sim near -1
                labels = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
                labels_diag = labels.diagonal()  # (B,) same-class indicator

                align = F.mse_loss(sim, 2 * labels_diag - 1)
                total_align += align
                pairs += 1

        return total_align / max(pairs, 1)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        modality_features: Optional[list] = None,
        model_parameters: Optional[list] = None
    ) -> dict:
        """
        Compute total loss.

        Args:
            logits: (B, num_classes) raw output
            targets: (B,) ground truth labels
            modality_features: List of modality feature tensors for alignment loss
            model_parameters: Model parameters for L2 regularization

        Returns:
            dict with 'total', 'classification', 'regularization', 'alignment'
        """
        # L_classification
        l_cls = self.classification_loss(logits, targets)

        # L_regularization (L2 norm)
        l_reg = torch.tensor(0.0, device=logits.device)
        if model_parameters is not None and self.lambda_reg > 0:
            for param in model_parameters:
                l_reg += param.pow(2).sum()
            l_reg = l_reg * 0.5

        # L_alignment
        l_align = torch.tensor(0.0, device=logits.device)
        if modality_features is not None and self.mu_align > 0:
            l_align = self.alignment_loss(modality_features, targets)

        # Total (Eq. 9)
        total = l_cls + self.lambda_reg * l_reg + self.mu_align * l_align

        return {
            "total": total,
            "classification": l_cls,
            "regularization": l_reg,
            "alignment": l_align
        }


if __name__ == "__main__":
    loss_fn = MultimodalLoss(lambda_reg=1e-4, mu_align=0.1, num_classes=1)
    logits = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8,))
    feats = [torch.randn(8, 256) for _ in range(3)]

    losses = loss_fn(logits, targets, modality_features=feats)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
