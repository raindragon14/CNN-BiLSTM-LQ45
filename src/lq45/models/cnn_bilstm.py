"""Arsitektur CNN-BiLSTM untuk prediksi imbal hasil.

Sumber pilihan arsitektur (rincian: docs/keputusan_desain.md):
- CNN dua lapis 32 -> 64: Espiga-Fernandez et al. (2024) Tabel 3.
- Kernel 3: Chaweewanchon & Chaysiri (2022) memakai konvolusi 3x3.
- BiLSTM dua lapis: Graves, Mohamed & Hinton (2013) memperkenalkan
  RNN berlapis; Chaweewanchon & Chaysiri (2022) dan Sebastian & Tantia
  (2024) memakai dua lapis LSTM.
- BatchNorm: Chaweewanchon & Chaysiri (2022) memasang BatchNorm pada
  arsitekturnya.
- Dropout 0,2: Sebastian & Tantia (2024) memakai nilai 0,2.
- ReLU: Espiga-Fernandez et al. (2024) Tabel 3 (aktivasi lapis konvolusi
  dan lapis Linear).

Pilihan implementasi yang tidak ditetapkan artikel dan dicatat di
docs/keputusan_desain.md: padding `same` pada konvolusi, urutan
Conv -> BatchNorm -> ReLU, dan inisialisasi bawaan PyTorch.

Versi 2: Encoder dipisah ke `encoder.py` untuk pre-training.
`CNNBiLSTM` sekarang wrapper tipis: `CNNBiLSTMEncoder` + `Linear(1)`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from lq45.models.encoder import CNNBiLSTMEncoder


class CNNBiLSTM(nn.Module):
    """CNN-BiLSTM lengkap dengan projection head untuk prediksi supervised.

    Wrapper: `CNNBiLSTMEncoder` + `Linear(1)`.
    API identik dengan versi sebelumnya untuk backward compatibility.
    """

    def __init__(
        self,
        n_features: int = 8,
        filters: Sequence[int] = (32, 64),
        kernel_size: int = 3,
        pooling: int = 2,
        units: int = 64,
        layers: int = 2,
        batchnorm: bool = True,
        dropout_cnn: float = 0.2,
        dropout_lstm: float = 0.2,
        dropout_dense: float = 0.2,
        bidirectional: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.encoder = CNNBiLSTMEncoder(
            n_features=n_features,
            filters=filters,
            kernel_size=kernel_size,
            pooling=pooling,
            units=units,
            layers=layers,
            batchnorm=batchnorm,
            dropout_cnn=dropout_cnn,
            dropout_lstm=dropout_lstm,
            bidirectional=bidirectional,
            activation=activation,
        )
        self.dropout_dense = nn.Dropout(dropout_dense)
        arah = 2 if bidirectional else 1
        self.dense = nn.Linear(units * arah, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Terapkan encoder lalu projection head."""
        z = self.encoder(x, return_sequence=False)  # (B, H)
        z = self.dropout_dense(z)
        return self.dense(z).squeeze(-1)

    def load_pretrained_encoder(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load bobot pre-trained encoder."""
        self.encoder.load_state_dict(state_dict)

    def freeze_encoder(self) -> None:
        """Bekukan parameter encoder untuk fine-tuning head saja."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder untuk full fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    @property
    def encoder_output_dim(self) -> int:
        return self.encoder.output_dim
