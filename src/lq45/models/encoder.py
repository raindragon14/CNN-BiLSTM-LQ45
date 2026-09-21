"""CNN-BiLSTM Encoder untuk pre-training dan fine-tuning.

Diekstrak dari CNNBiLSTM agar bisa dipakai sebagai backbone terpisah.
Arsitektur identik dengan CNNBiLSTM.forward() hingga output BiLSTM.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class CNNBiLSTMEncoder(nn.Module):
    """Encoder CNN dua lapis + BiLSTM dua lapis.

    Masukan: (B, F, w) mengikuti konvensi Conv1d.
    Keluaran:
      - return_sequence=True:  (B, L, H) urutan penuh BiLSTM
      - return_sequence=False: (B, H)   langkah terakhir (untuk fine-tune head)
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
            raise ValueError(f"aktivasi tidak didukung: {activation}")

        # CNN blocks
        blok: list[nn.Module] = []
        masuk = n_features
        for keluar in filters:
            padding = kernel_size // 2  # 'same'
            blok.append(nn.Conv1d(masuk, keluar, kernel_size, padding=padding))
            if batchnorm:
                blok.append(nn.BatchNorm1d(keluar))
            blok.append(nn.ReLU())
            masuk = keluar
        self.conv = nn.Sequential(*blok)
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
        """Terapkan CNN -> Pool -> Dropout -> BiLSTM.

        Args:
            x: (B, F, w)
            return_sequence: True -> (B, L, H*dir), False -> (B, H*dir)

        Returns:
            Tensor sesuai return_sequence.
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
        """Dimensi output per timestep (H * 2 jika bidirectional)."""
        arah = 2 if self.bidirectional else 1
        return self.units * arah
