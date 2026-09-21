"""Decoder rekonstruksi untuk Masked Autoencoder (MAE).

Menerima latent sequence dari encoder, upsampling kembali ke panjang
lookback asli, lalu memprediksi 5 channel OHLCV per timestep.
"""

from __future__ import annotations

import torch
from torch import nn


class MAEDecoder(nn.Module):
    """Decoder ConvTranspose1d + Linear untuk merekonstruksi OHLCV.

    Arsitektur:
      - Input: (B, L, H) dari encoder (L = w // pooling)
      - ConvTranspose1d layers untuk upsampling ke panjang lookback
      - Linear layer final: hidden -> 5 channel (OHLCV)
    """

    def __init__(
        self,
        latent_dim: int,
        n_channels: int = 5,  # OHLCV target
        lookback: int = 60,
        pooling: int = 2,
        hidden: int = 128,
        layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation != "relu":
            raise ValueError(f"aktivasi tidak didukung: {activation}")

        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.lookback = lookback
        self.pooling = pooling
        self.seq_len = lookback // pooling  # panjang sequence setelah pooling

        # Proyeksi latent_dim -> hidden
        self.proj = nn.Linear(latent_dim, hidden)

        # ConvTranspose1d untuk upsampling
        # seq_len -> lookback via strided transpose conv
        blok: list[nn.Module] = []
        in_ch = hidden
        current_len = self.seq_len

        # Hitung strides yang dibutuhkan untuk naik dari seq_len ke lookback
        # Gunakan pooling factor sebagai stride utama
        stride = pooling
        kernel = stride * 2
        padding = stride // 2
        output_padding = 0

        for i in range(layers):
            out_ch = hidden
            blok.append(
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            if i < layers - 1:
                blok.append(nn.BatchNorm1d(out_ch))
                blok.append(nn.ReLU())
            in_ch = out_ch
            current_len = (
                (current_len - 1) * stride - 2 * padding + kernel + output_padding
            )

        # Jika panjang belum pas, tambah layer penyesuaian
        if current_len != lookback:
            # Layer 1x1 conv untuk penyesuaian channel, lalu interpolate
            blok.append(nn.Conv1d(hidden, hidden, 1))
            self.need_interpolate = True
            self.target_len = lookback
        else:
            self.need_interpolate = False

        self.upsample = nn.Sequential(*blok)
        self.final = nn.Linear(hidden, n_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct OHLCV dari latent sequence.

        Args:
            z: (B, L, H) dari encoder dengan return_sequence=True

        Returns:
            (B, n_channels, lookback) rekonstruksi OHLCV
        """
        _, _, _ = z.shape
        # Proyeksi ke hidden
        z = self.proj(z)  # (B, L, hidden)
        z = z.permute(0, 2, 1)  # (B, hidden, L)
        z = self.upsample(z)  # (B, hidden, ~lookback)

        if self.need_interpolate:
            z = torch.nn.functional.interpolate(
                z, size=self.target_len, mode="linear", align_corners=False
            )

        z = z.permute(0, 2, 1)  # (B, lookback, hidden)
        out = self.final(z)  # (B, lookback, n_channels)
        return out.permute(0, 2, 1)  # (B, n_channels, lookback)
