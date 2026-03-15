"""
Cross-Modal Attention Fusion Module
Implements intermediate fusion with dynamic modality weighting.
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"

Core equations from the paper:
    F_fusion = sum_m(alpha_m * F_m)                          [Eq. 5]
    alpha_m = softmax(score_m)                               [Eq. 6]
    F_fusion = sum_{m in M}(alpha_m * F_m)  [with dropout]  [Eq. 7]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention mechanism that dynamically weights modality contributions.

    For each modality, computes an attention score by projecting its feature vector
    through a shared scoring network. The scores are normalized via softmax across
    modalities, producing alpha weights that reflect each modality's relevance.

    Args:
        latent_dim (int): Dimension of each modality's feature vector
        num_modalities (int): Number of modalities (default: 3)
        num_heads (int): Number of attention heads for multi-head variant
        dropout (float): Dropout on attention weights
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_modalities: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(CrossModalAttention, self).__init__()

        self.latent_dim = latent_dim
        self.num_modalities = num_modalities

        # Scoring network: maps each modality feature -> scalar score
        self.score_network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Multi-head cross-modal transformer attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality_features: List[torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-modal attention-weighted fusion.

        Args:
            modality_features (List[Tensor]): List of M feature tensors, each (B, latent_dim)
                                              Missing modalities should be replaced with zeros.
            modality_mask (Tensor, optional): Boolean mask (B, M) — True = modality is ABSENT
                                              Used to zero out missing modality contributions.

        Returns:
            dict with keys:
                'fused'   : (B, latent_dim) — fused multimodal representation
                'weights' : (B, M) — per-modality attention weights (for interpretability)
        """
        B = modality_features[0].shape[0]
        M = len(modality_features)

        # Stack modality features: (B, M, latent_dim)
        stacked = torch.stack(modality_features, dim=1)

        # ---- Scalar attention weights (Eq. 5, 6 from paper) ----
        scores = self.score_network(stacked)       # (B, M, 1)
        scores = scores.squeeze(-1)                # (B, M)

        # Mask out absent modalities with large negative value
        if modality_mask is not None:
            scores = scores.masked_fill(modality_mask, float('-inf'))

        alpha = F.softmax(scores, dim=-1)          # (B, M) — attention weights

        # Handle NaN when all modalities masked (shouldn't happen in practice)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Weighted sum (Eq. 5): F_fusion = sum_m(alpha_m * F_m)
        alpha_expanded = alpha.unsqueeze(-1)       # (B, M, 1)
        weighted = stacked * alpha_expanded        # (B, M, latent_dim)
        fused_scalar = weighted.sum(dim=1)         # (B, latent_dim)

        # ---- Multi-head cross-modal attention (richer interaction) ----
        # Query: all modalities attend to all modalities
        key_padding_mask = modality_mask if modality_mask is not None else None
        attn_out, _ = self.multihead_attn(
            query=stacked,
            key=stacked,
            value=stacked,
            key_padding_mask=key_padding_mask
        )  # (B, M, latent_dim)

        # Residual connection + norm
        attn_out = self.norm(stacked + self.dropout(attn_out))  # (B, M, latent_dim)

        # Pool over modality dimension using alpha weights
        fused_multihead = (attn_out * alpha_expanded).sum(dim=1)  # (B, latent_dim)

        # Final fusion: combine both paths
        fused = fused_scalar + fused_multihead

        return {"fused": fused, "weights": alpha}


class ModalityDropout(nn.Module):
    """
    Modality Dropout for robustness training.

    During training, randomly zeros out one or more modality features
    to simulate real-world missing data scenarios.
    Forces the model to learn compensatory patterns using available modalities.

    Args:
        num_modalities (int): Total number of modalities
        dropout_prob (float): Probability of dropping any single modality per sample
        min_active (int): Minimum number of modalities that must remain active
    """

    def __init__(
        self,
        num_modalities: int = 3,
        dropout_prob: float = 0.3,
        min_active: int = 1
    ):
        super(ModalityDropout, self).__init__()
        self.num_modalities = num_modalities
        self.dropout_prob = dropout_prob
        self.min_active = min_active

    def forward(
        self,
        modality_features: List[torch.Tensor]
    ) -> tuple:
        """
        Args:
            modality_features: List of M tensors, each (B, latent_dim)
        Returns:
            dropped_features: List of M tensors (some zeroed out)
            mask: (B, M) boolean tensor — True where modality was dropped
        """
        B = modality_features[0].shape[0]
        M = len(modality_features)

        if not self.training:
            # No dropout at inference time
            mask = torch.zeros(B, M, dtype=torch.bool, device=modality_features[0].device)
            return modality_features, mask

        # Sample dropout mask per sample in batch
        mask = torch.rand(B, M, device=modality_features[0].device) < self.dropout_prob

        # Ensure at least min_active modalities remain per sample
        active_count = (~mask).sum(dim=1, keepdim=True)  # (B, 1)
        insufficient = active_count < self.min_active     # (B, 1)

        if insufficient.any():
            # For samples with too few active modalities, randomly activate one
            random_activate = torch.randint(0, M, (B,), device=mask.device)
            for b in range(B):
                if insufficient[b]:
                    mask[b, random_activate[b]] = False

        # Apply mask to features (zero out dropped modalities)
        dropped = []
        for m_idx, feat in enumerate(modality_features):
            m_mask = mask[:, m_idx].unsqueeze(-1).float()  # (B, 1)
            dropped.append(feat * (1.0 - m_mask))

        return dropped, mask


if __name__ == "__main__":
    # Test cross-modal attention
    B, latent_dim = 4, 256
    img_feat = torch.randn(B, latent_dim)
    ecg_feat = torch.randn(B, latent_dim)
    ehr_feat = torch.randn(B, latent_dim)

    attn = CrossModalAttention(latent_dim=256, num_modalities=3, num_heads=4)
    result = attn([img_feat, ecg_feat, ehr_feat])
    print(f"Fused shape:   {result['fused'].shape}")    # (4, 256)
    print(f"Weights shape: {result['weights'].shape}")  # (4, 3)
    print(f"Weights (sum): {result['weights'].sum(dim=1)}")  # Should be ~1.0

    # Test modality dropout
    dropper = ModalityDropout(num_modalities=3, dropout_prob=0.4, min_active=1)
    dropper.train()
    dropped, mask = dropper([img_feat, ecg_feat, ehr_feat])
    print(f"\nDropout mask:\n{mask}")
    print(f"Active modalities per sample: {(~mask).sum(dim=1)}")
