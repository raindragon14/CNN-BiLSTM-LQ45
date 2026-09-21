"""Masked Autoencoder (MAE) pre-training untuk CNN-BiLSTM.

Pre-training self-supervised pada data OHLCV + makro (7 channel)
tanpa label return. Encoder belajar representasi harga universal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lq45.models.decoder import MAEDecoder
from lq45.models.encoder import CNNBiLSTMEncoder
from lq45.models.training import set_seed


@dataclass
class PretrainResult:
    """Hasil pre-training MAE."""

    encoder_state: dict[str, Any]
    decoder_state: dict[str, Any]
    history: list[dict[str, float]]
    best_val_loss: float
    best_epoch: int
    wall_sec: float


def mask_input(
    x: torch.Tensor,
    mask_ratio: float = 0.3,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masking acak pada dimensi waktu.

    Args:
        x: (B, C, T) input tensor
        mask_ratio: fraksi timestep yang di-mask
        generator: RNG untuk reproducibility

    Returns:
        x_masked: (B, C, T) dengan posisi masked = 0
        mask: (B, T) boolean, True = posisi yang di-mask
    """
    B, _, T = x.shape
    n_mask = int(T * mask_ratio)

    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for b in range(B):
        idx = torch.randperm(T, generator=generator, device=x.device)[:n_mask]
        mask[b, idx] = True

    x_masked = x.clone()
    x_masked[:, :, mask.any(dim=0)] = 0  # zero-out masked timesteps across all channels
    return x_masked, mask


def mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss hanya pada posisi yang di-mask.

    Args:
        pred: (B, C, T) prediksi decoder
        target: (B, C, T) target asli
        mask: (B, T) boolean mask

    Returns:
        Scalar loss
    """
    # Expand mask ke channel dimension
    mask_expanded = mask.unsqueeze(1).expand_as(pred)  # (B, C, T)
    diff = (pred - target) ** 2
    masked_diff = diff[mask_expanded]
    if masked_diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return masked_diff.mean()


def build_pretrain_data(
    panel: Any,
    lookback: int,
    pretrain_channels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Bangun data pre-train dari panel OHLCV + makro.

    Args:
        panel: PanelData dengan features mentah
        lookback: panjang jendela
        pretrain_channels: list nama channel untuk pre-train (7 channel)

    Returns:
        X: (N, C, lookback) tanpa label
        dates_idx: (N,) posisi tanggal
    """
    # Indeks channel yang dipakai untuk pre-train
    from lq45.models.dataset import FEATURE_COLUMNS, build_windows

    channel_idx = [FEATURE_COLUMNS.index(c) for c in pretrain_channels]

    daftar_x: list[np.ndarray] = []
    daftar_idx: list[np.ndarray] = []

    for ticker in panel.tickers:
        feat = panel.features[ticker][:, channel_idx]  # (T, 7)
        # Gunakan seluruh timeline tanpa label, tanpa banned
        x, _, idx = build_windows(
            feat,
            np.zeros(len(feat)),  # dummy target
            lookback,
            0,
            len(feat),
            [],
        )
        if len(x):
            daftar_x.append(x)
            daftar_idx.append(idx)

    if not daftar_x:
        raise ValueError("tidak ada data pre-train")

    X = np.concatenate(daftar_x, axis=0)
    idx = np.concatenate(daftar_idx, axis=0)
    return X, idx


def pretrain_mae(
    encoder: CNNBiLSTMEncoder,
    decoder: MAEDecoder,
    train_x: np.ndarray,
    val_x: np.ndarray,
    mask_ratio: float = 0.3,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    patience: int = 10,
    device: torch.device | None = None,
    seed: int = 0,
) -> PretrainResult:
    """Latih MAE: encoder + decoder merekonstruksi timestep yang di-mask.

    Args:
        encoder: CNNBiLSTMEncoder
        decoder: MAEDecoder
        train_x: (N, C, lookback) data latih
        val_x: (N, C, lookback) data validasi
        mask_ratio: fraksi masking
        epochs: max epoch
        lr: learning rate
        weight_decay: weight decay
        batch_size: batch size
        patience: early stopping patience
        device: torch device
        seed: random seed

    Returns:
        PretrainResult dengan best encoder/decoder state
    """
    if device is None:
        device = torch.device("cpu")
    generator = set_seed(seed)

    train_ds = TensorDataset(torch.from_numpy(train_x))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator
    )

    encoder.to(device)
    decoder.to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_epoch = 0
    encoder_best = {}
    decoder_best = {}
    wait = 0
    history: list[dict[str, float]] = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        train_losses = []

        for (xb,) in train_loader:
            xb = xb.to(device)  # (B, 7, T)
            xb_masked, mask = mask_input(xb, mask_ratio, generator)

            optimizer.zero_grad()
            z = encoder(xb_masked, return_sequence=True)  # (B, L, H)
            pred = decoder(z)  # (B, 5, T)
            # Compute loss only on OHLCV channels (first 5)
            loss = mae_loss(pred, xb[:, :5, :], mask)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # Validasi
        encoder.eval()
        decoder.eval()
        val_losses = []
        with torch.no_grad():
            val_ds = TensorDataset(torch.from_numpy(val_x))
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            for (xb,) in val_loader:
                xb = xb.to(device)
                xb_masked, mask = mask_input(xb, mask_ratio, generator)
                z = encoder(xb_masked, return_sequence=True)
                pred = decoder(z)
                # Compute loss only on OHLCV channels (first 5)
                loss = mae_loss(pred, xb[:, :5, :], mask)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        history.append(
            {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            encoder_best = {k: v.clone() for k, v in encoder.state_dict().items()}
            decoder_best = {k: v.clone() for k, v in decoder.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    encoder.load_state_dict(encoder_best)
    decoder.load_state_dict(decoder_best)

    return PretrainResult(
        encoder_state=encoder_best,
        decoder_state=decoder_best,
        history=history,
        best_val_loss=best_val,
        best_epoch=best_epoch,
        wall_sec=time.time() - start,
    )


def load_pretrained_encoder(
    encoder: CNNBiLSTMEncoder,
    state_dict: dict[str, Any],
) -> None:
    """Load pretrained weights ke encoder."""
    encoder.load_state_dict(state_dict)
