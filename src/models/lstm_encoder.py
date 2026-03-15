"""
Bi-LSTM / GRU Encoder for Sequential Physiological Data (ECG, EEG)
Captures temporal dependencies in time-series signals.
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM / GRU encoder for sequential physiological signals.

    Processes fixed-length windows of time-series data (e.g., 60-second ECG segments)
    and extracts temporal features via hidden state pooling.

    Args:
        input_dim (int): Number of input channels/features per timestep
        hidden_dim (int): LSTM hidden state dimension
        num_layers (int): Number of stacked LSTM layers
        latent_dim (int): Output dimension for shared latent space
        rnn_type (str): 'lstm' or 'gru'
        bidirectional (bool): Use bidirectional RNN
        dropout (float): Dropout between RNN layers
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 256,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super(LSTMEncoder, self).__init__()

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # Input projection: normalize raw signal -> learnable embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Recurrent layers
        rnn_class = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        rnn_output_dim = hidden_dim * self.num_directions

        # Attention pooling over time steps
        self.temporal_attention = nn.Sequential(
            nn.Linear(rnn_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Projection to shared latent space
        self.projection = nn.Sequential(
            nn.Linear(rnn_output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )

    def attention_pool(self, rnn_out: torch.Tensor) -> torch.Tensor:
        """
        Weighted average pooling over time dimension using learned attention.

        Args:
            rnn_out: (B, T, rnn_output_dim)
        Returns:
            pooled: (B, rnn_output_dim)
        """
        scores = self.temporal_attention(rnn_out)      # (B, T, 1)
        weights = torch.softmax(scores, dim=1)          # (B, T, 1)
        pooled = (rnn_out * weights).sum(dim=1)         # (B, rnn_output_dim)
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input time-series of shape (B, T, input_dim)
                               B=batch, T=timesteps (e.g. 1500 for 60s at 25Hz)
            lengths (torch.Tensor, optional): Actual sequence lengths for padding (B,)

        Returns:
            torch.Tensor: Latent feature vector (B, latent_dim)
        """
        B, T, _ = x.shape

        # Project input features
        x_proj = self.input_proj(x)   # (B, T, hidden_dim)

        # Pack padded sequence for efficiency (if lengths provided)
        if lengths is not None:
            x_proj = nn.utils.rnn.pack_padded_sequence(
                x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # Run through RNN
        if self.rnn_type == "lstm":
            rnn_out, (h_n, _) = self.rnn(x_proj)
        else:
            rnn_out, h_n = self.rnn(x_proj)

        # Unpack if needed
        if lengths is not None:
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # Attention-based temporal pooling
        pooled = self.attention_pool(rnn_out)   # (B, rnn_output_dim)

        # Project to latent space
        latent = self.projection(pooled)         # (B, latent_dim)
        return latent


if __name__ == "__main__":
    # Test with ECG-like input: batch=4, 1500 timesteps, 1 channel (lead II)
    encoder = LSTMEncoder(
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        latent_dim=256,
        rnn_type="lstm",
        bidirectional=True
    )
    dummy_ecg = torch.randn(4, 1500, 1)
    output = encoder(dummy_ecg)
    print(f"LSTM Encoder output shape: {output.shape}")  # Expected: (4, 256)
    print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
