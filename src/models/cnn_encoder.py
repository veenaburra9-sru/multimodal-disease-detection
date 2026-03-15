"""
CNN Encoder for Medical Imaging (X-ray, CT, MRI)
Uses pretrained ResNet-50 or EfficientNet as backbone.
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """
    CNN-based encoder for radiological images.
    Extracts spatial hierarchical features using a pretrained backbone.

    Supports:
        - ResNet-50
        - EfficientNet-B0 to B7

    Args:
        backbone (str): One of ['resnet50', 'efficientnet_b0', ..., 'efficientnet_b4']
        latent_dim (int): Output feature dimension for shared latent space
        pretrained (bool): Whether to use ImageNet pretrained weights
        freeze_backbone (bool): Freeze backbone weights (for fine-tuning only top layers)
    """

    SUPPORTED_BACKBONES = [
        "resnet50", "resnet18", "resnet101",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "efficientnet_b3", "efficientnet_b4"
    ]

    def __init__(
        self,
        backbone: str = "resnet50",
        latent_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super(CNNEncoder, self).__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone '{backbone}' not supported. Choose from: {self.SUPPORTED_BACKBONES}")

        self.backbone_name = backbone
        self.latent_dim = latent_dim

        # Load backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone, feature_dim = self._build_backbone(backbone, weights)

        # Freeze backbone if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head: maps backbone features -> shared latent space
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )

    def _build_backbone(self, backbone: str, weights):
        """Returns (backbone_model_without_classifier, feature_dim)."""
        if backbone.startswith("resnet"):
            model_fn = getattr(models, backbone)
            model = model_fn(weights=weights)
            feature_dim = model.fc.in_features
            # Remove final classification layer; use GAP output
            model.fc = nn.Identity()
            return model, feature_dim

        elif backbone.startswith("efficientnet"):
            model_fn = getattr(models, backbone)
            model = model_fn(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, feature_dim

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
                              Expected: (B, 3, 224, 224)
        Returns:
            torch.Tensor: Latent feature vector of shape (B, latent_dim)
        """
        features = self.backbone(x)          # (B, feature_dim)
        latent = self.projection(features)   # (B, latent_dim)
        return latent


if __name__ == "__main__":
    # Quick test
    encoder = CNNEncoder(backbone="resnet50", latent_dim=256, pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 X-ray images
    output = encoder(dummy_input)
    print(f"CNN Encoder output shape: {output.shape}")  # Expected: (4, 256)
    print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
