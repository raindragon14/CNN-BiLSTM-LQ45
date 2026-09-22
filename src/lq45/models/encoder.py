"""CNN-BiLSTM encoder for pre-training and fine-tuning.

Extracted from CNNBiLSTM so it can be used as a standalone backbone.
Architecture identical to CNNBiLSTM.forward() up to the BiLSTM output.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class CNNBiLSTMEncoder(nn.Module):
    """Two-layer CNN encoder + two-layer BiLSTM.

    Input: (B, F, w) following the Conv1d convention.
    Output:
      - return_sequence=True:  (B, L, H) full BiLSTM sequence
      - return_sequence=False: (B, H)   last step (for fine-tuning the head)
    """

    def __init__(
        self,
        n_features: int = 7,
        filters: Sequence[int] = (32, 64),
        kernel_size: int = 3,
        pooling: int = 2,
        units: int = 64,
        layers: int = 2,
        batchnorm: bool = True,
        dropout_cnn: float = 0.2,
        dropout_lstm: float = 0.2,
        bidirectional: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation != "relu":
            raise ValueError(f"unsupported activation: {activation}")

        # CNN blocks
        blocks: list[nn.Module] = []
        in_channels = n_features
        for out_channels in filters:
            padding = kernel_size // 2  # 'same'
            blocks.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            )
            if batchnorm:
                blocks.append(nn.BatchNorm1d(out_channels))
            blocks.append(nn.ReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.MaxPool1d(pooling)
        self.dropout_cnn = nn.Dropout(dropout_cnn)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=filters[-1],
            hidden_size=units,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_lstm if layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional
        self.units = units
        self.layers = layers

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """Apply CNN -> Pool -> Dropout -> BiLSTM.

        Args:
            x: (B, F, w)
            return_sequence: True -> (B, L, H*dir), False -> (B, H*dir)

        Returns:
            Tensor according to `return_sequence`.
        """
        x = self.conv(x)  # (B, C, w)
        x = self.pool(x)  # (B, C, w // p)
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1)  # (B, L, C)
        x, _ = self.lstm(x)  # (B, L, H*dir)
        if return_sequence:
            return x
        return x[:, -1, :]  # (B, H*dir)

    @property
    def output_dim(self) -> int:
        """Output dimension per timestep (H * 2 if bidirectional)."""
        directions = 2 if self.bidirectional else 1
        return self.units * directions
