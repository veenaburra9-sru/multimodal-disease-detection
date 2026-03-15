"""
Full Multimodal Model: Integrates CNN + Bi-LSTM + MLP with Cross-Modal Attention
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"

y = sigma(W_pred * F_fusion + b_pred)  [Eq. 8]
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .cnn_encoder import CNNEncoder
from .lstm_encoder import LSTMEncoder
from .mlp_encoder import MLPEncoder
from .cross_modal_attention import CrossModalAttention, ModalityDropout


class MultimodalDiseaseDetector(nn.Module):
    """
    Tri-modal deep learning model for early disease detection.

    Three processing branches:
      1. CNN branch    — processes radiological images (X-ray, CT)
      2. LSTM branch   — processes sequential physiological signals (ECG, EEG)
      3. MLP branch    — processes structured EHR data (lab results, demographics)

    Fusion via intermediate cross-modal attention (Eq. 5–7 from paper).
    Decision head outputs disease probability (Eq. 8 from paper).

    Args:
        num_classes (int): Number of output classes (1 for binary, >1 for multi-class)
        latent_dim (int): Shared latent space dimension for all modalities
        img_backbone (str): CNN backbone name ('resnet50', 'efficientnet_b0', ...)
        ecg_input_dim (int): Input features per ECG timestep
        ehr_input_dim (int): Number of structured EHR features
        modality_dropout_prob (float): Probability of dropping a modality during training
        num_attention_heads (int): Number of heads in cross-modal attention
        freeze_cnn (bool): Freeze CNN backbone weights
    """

    def __init__(
        self,
        num_classes: int = 1,
        latent_dim: int = 256,
        img_backbone: str = "resnet50",
        ecg_input_dim: int = 1,
        ehr_input_dim: int = 64,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        ehr_hidden_dims=None,
        modality_dropout_prob: float = 0.3,
        num_attention_heads: int = 4,
        classifier_dropout: float = 0.4,
        freeze_cnn: bool = False,
        pretrained_cnn: bool = True
    ):
        super(MultimodalDiseaseDetector, self).__init__()

        if ehr_hidden_dims is None:
            ehr_hidden_dims = [256, 256]

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # ── Modality Encoders ────────────────────────────────────────
        self.cnn_encoder = CNNEncoder(
            backbone=img_backbone,
            latent_dim=latent_dim,
            pretrained=pretrained_cnn,
            freeze_backbone=freeze_cnn
        )

        self.lstm_encoder = LSTMEncoder(
            input_dim=ecg_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            latent_dim=latent_dim,
            rnn_type="lstm",
            bidirectional=True
        )

        self.mlp_encoder = MLPEncoder(
            input_dim=ehr_input_dim,
            hidden_dims=ehr_hidden_dims,
            latent_dim=latent_dim
        )

        # ── Modality Dropout (Eq. 7) ─────────────────────────────────
        self.modality_dropout = ModalityDropout(
            num_modalities=3,
            dropout_prob=modality_dropout_prob,
            min_active=1
        )

        # ── Cross-Modal Attention Fusion (Eq. 5, 6) ──────────────────
        self.fusion = CrossModalAttention(
            latent_dim=latent_dim,
            num_modalities=3,
            num_heads=num_attention_heads
        )

        # ── Decision Head (Eq. 8) ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize classifier and projection layers with Xavier uniform."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_modalities(
        self,
        images: Optional[torch.Tensor] = None,
        ecg: Optional[torch.Tensor] = None,
        ehr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode each modality independently.

        Returns:
            img_feat, ecg_feat, ehr_feat: each (B, latent_dim) — zeros if modality absent
            modality_missing_mask: (B, 3) — True where a modality is None/missing
        """
        B = self._get_batch_size(images, ecg, ehr)
        device = self._get_device(images, ecg, ehr)

        # Track which modalities are structurally absent (not just dropped)
        absent_mask = torch.zeros(B, 3, dtype=torch.bool, device=device)

        if images is not None:
            img_feat = self.cnn_encoder(images)
        else:
            img_feat = torch.zeros(B, self.latent_dim, device=device)
            absent_mask[:, 0] = True

        if ecg is not None:
            ecg_feat = self.lstm_encoder(ecg)
        else:
            ecg_feat = torch.zeros(B, self.latent_dim, device=device)
            absent_mask[:, 1] = True

        if ehr is not None:
            ehr_feat = self.mlp_encoder(ehr)
        else:
            ehr_feat = torch.zeros(B, self.latent_dim, device=device)
            absent_mask[:, 2] = True

        return img_feat, ecg_feat, ehr_feat, absent_mask

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        ecg: Optional[torch.Tensor] = None,
        ehr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images (Tensor, optional): (B, 3, 224, 224) — chest X-ray images
            ecg    (Tensor, optional): (B, T, 1) — ECG time-series
            ehr    (Tensor, optional): (B, ehr_input_dim) — structured EHR features

        Returns:
            dict with keys:
                'logits'          : (B, num_classes) — raw prediction scores
                'probabilities'   : (B, num_classes) — sigmoid/softmax probabilities
                'attention_weights': (B, 3) — per-modality weights (imaging, ECG, EHR)
                'fused_features'  : (B, latent_dim) — fused representation
        """
        # 1. Encode each modality
        img_feat, ecg_feat, ehr_feat, absent_mask = self.encode_modalities(images, ecg, ehr)

        # 2. Modality dropout (training only — Eq. 7)
        features, dropout_mask = self.modality_dropout([img_feat, ecg_feat, ehr_feat])

        # Combined mask: absent OR dropped
        combined_mask = absent_mask | dropout_mask

        # 3. Cross-modal attention fusion (Eq. 5, 6)
        fusion_result = self.fusion(features, modality_mask=combined_mask)
        fused = fusion_result["fused"]          # (B, latent_dim)
        attn_weights = fusion_result["weights"] # (B, 3)

        # 4. Classification head (Eq. 8)
        logits = self.classifier(fused)         # (B, num_classes)

        if self.num_classes == 1:
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "attention_weights": attn_weights,
            "fused_features": fused
        }

    def _get_batch_size(self, *tensors) -> int:
        for t in tensors:
            if t is not None:
                return t.shape[0]
        raise ValueError("At least one modality must be provided.")

    def _get_device(self, *tensors) -> torch.device:
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        return {
            "cnn_encoder": sum(p.numel() for p in self.cnn_encoder.parameters()),
            "lstm_encoder": sum(p.numel() for p in self.lstm_encoder.parameters()),
            "mlp_encoder": sum(p.numel() for p in self.mlp_encoder.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
            "total": sum(p.numel() for p in self.parameters())
        }


if __name__ == "__main__":
    # Test full model
    model = MultimodalDiseaseDetector(
        num_classes=1,
        latent_dim=256,
        img_backbone="resnet50",
        ecg_input_dim=1,
        ehr_input_dim=64,
        pretrained_cnn=False
    )

    B = 4
    images = torch.randn(B, 3, 224, 224)
    ecg    = torch.randn(B, 1500, 1)
    ehr    = torch.randn(B, 64)

    model.eval()
    with torch.no_grad():
        # Tri-modal
        out = model(images=images, ecg=ecg, ehr=ehr)
        print("=== Tri-modal ===")
        print(f"  Logits shape:       {out['logits'].shape}")
        print(f"  Probabilities:      {out['probabilities'].squeeze()}")
        print(f"  Attention weights:  {out['attention_weights']}")

        # Missing imaging modality
        out2 = model(ecg=ecg, ehr=ehr)
        print("\n=== Missing Imaging ===")
        print(f"  Probabilities: {out2['probabilities'].squeeze()}")

    params = model.count_parameters()
    print(f"\nParameter counts: {params}")
