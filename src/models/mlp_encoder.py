"""
MLP Encoder for Structured Clinical Data (EHR, Lab Results, Demographics)
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"
"""

import torch
import torch.nn as nn
from typing import List


class MLPEncoder(nn.Module):
    """
    Multilayer Perceptron encoder for structured clinical/tabular data.

    Handles heterogeneous inputs such as:
      - Lab results (troponin, CRP, creatinine, ...)
      - Patient demographics (age, sex, BMI, ...)
      - Vital signs (heart rate, BP, O2 saturation, ...)
      - ICD codes (as embeddings)

    Applies BatchNorm + Dropout for regularization.

    Args:
        input_dim (int): Number of structured input features
        hidden_dims (List[int]): Sizes of hidden layers
        latent_dim (int): Output dimension for shared latent space
        dropout (float): Dropout probability
        use_batch_norm (bool): Apply BatchNorm after each layer
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: List[int] = [256, 256],
        latent_dim: int = 256,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(MLPEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build sequential MLP layers
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # Projection to shared latent space
        self.projection = nn.Sequential(
            nn.Linear(prev_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Structured feature vector (B, input_dim)
                               Should be preprocessed: normalized continuous,
                               one-hot encoded categorical, imputed missing values.
        Returns:
            torch.Tensor: Latent feature vector (B, latent_dim)
        """
        hidden = self.mlp(x)           # (B, last_hidden_dim)
        latent = self.projection(hidden)  # (B, latent_dim)
        return latent


class EHREmbeddingEncoder(nn.Module):
    """
    Enhanced EHR encoder that handles mixed data types:
      - Continuous features (lab values, vitals) via linear projection
      - Categorical features (ICD codes, diagnoses) via embedding lookup

    Args:
        continuous_dim (int): Number of continuous input features
        categorical_vocab_sizes (List[int]): Vocabulary size for each categorical feature
        categorical_embed_dim (int): Embedding dimension for categorical features
        hidden_dims (List[int]): Hidden layer dimensions
        latent_dim (int): Output latent dimension
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        continuous_dim: int = 40,
        categorical_vocab_sizes: List[int] = None,
        categorical_embed_dim: int = 16,
        hidden_dims: List[int] = [256, 256],
        latent_dim: int = 256,
        dropout: float = 0.3
    ):
        super(EHREmbeddingEncoder, self).__init__()

        self.has_categorical = categorical_vocab_sizes is not None and len(categorical_vocab_sizes) > 0

        # Embeddings for categorical features
        if self.has_categorical:
            self.embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, categorical_embed_dim)
                for vocab_size in categorical_vocab_sizes
            ])
            total_cat_dim = categorical_embed_dim * len(categorical_vocab_sizes)
        else:
            total_cat_dim = 0

        input_dim = continuous_dim + total_cat_dim

        # Core MLP
        self.mlp_encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            continuous (torch.Tensor): (B, continuous_dim)
            categorical (torch.Tensor, optional): (B, num_categorical) — integer indices
        Returns:
            torch.Tensor: (B, latent_dim)
        """
        features = [continuous]

        if self.has_categorical and categorical is not None:
            cat_embeds = [
                emb(categorical[:, i])
                for i, emb in enumerate(self.embeddings)
            ]  # List of (B, embed_dim)
            features.append(torch.cat(cat_embeds, dim=-1))

        x = torch.cat(features, dim=-1)
        return self.mlp_encoder(x)


if __name__ == "__main__":
    # Test basic MLP encoder
    encoder = MLPEncoder(input_dim=64, hidden_dims=[256, 256], latent_dim=256)
    dummy_ehr = torch.randn(4, 64)
    output = encoder(dummy_ehr)
    print(f"MLP Encoder output shape: {output.shape}")  # Expected: (4, 256)

    # Test EHR embedding encoder (with categorical ICD codes)
    ehr_encoder = EHREmbeddingEncoder(
        continuous_dim=40,
        categorical_vocab_sizes=[1000, 500, 200],  # e.g., 3 categorical features
        latent_dim=256
    )
    cont = torch.randn(4, 40)
    cats = torch.randint(0, 100, (4, 3))
    output2 = ehr_encoder(cont, cats)
    print(f"EHR Embedding Encoder output shape: {output2.shape}")  # Expected: (4, 256)
