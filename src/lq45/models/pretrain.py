"""Masked Autoencoder (MAE) pre-training for CNN-BiLSTM.

Self-supervised pre-training on the processed feature panel
(FEATURE_COLUMNS) without return labels. The encoder input matches the
supervised model exactly, so the encoder weights transfer unchanged.
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
    """MAE pre-training result."""

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
    """Random masking along the time dimension.

    Args:
        x: (B, C, T) input tensor
        mask_ratio: fraction of timesteps to mask
        generator: RNG for reproducibility

    Returns:
        x_masked: (B, C, T) new copy; masked positions per sample = 0
        mask: (B, T) boolean, True = masked positions (per sample)
    """
    B, _, T = x.shape
    n_mask = int(T * mask_ratio)

    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for b in range(B):
        idx = torch.randperm(T, generator=generator, device=x.device)[:n_mask]
        mask[b, idx] = True

    x_masked = x.masked_fill(mask.unsqueeze(1), 0.0)  # (B,1,T) broadcast over channels
    return x_masked, mask


def mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss only at masked positions.

    Args:
        pred: (B, C, T) decoder predictions
        target: (B, C, T) original targets
        mask: (B, T) boolean mask

    Returns:
        Scalar loss
    """
    # Expand the mask to the channel dimension
    mask_expanded = mask.unsqueeze(1).expand_as(pred)  # (B, C, T)
    diff = (pred - target) ** 2
    masked_diff = diff[mask_expanded]
    if masked_diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return masked_diff.mean()


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
    """Train the MAE: encoder + decoder reconstruct the masked timesteps.

    Args:
        encoder: CNNBiLSTMEncoder
        decoder: MAEDecoder
        train_x: (N, C, lookback) training data
        val_x: (N, C, lookback) validation data
        mask_ratio: masking fraction
        epochs: max epochs
        lr: learning rate
        weight_decay: weight decay
        batch_size: batch size
        patience: early stopping patience
        device: torch device
        seed: random seed

    Returns:
        PretrainResult with the best encoder/decoder state
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
            xb = xb.to(device)
            xb_masked, mask = mask_input(xb, mask_ratio, generator)

            optimizer.zero_grad()
            z = encoder(xb_masked, return_sequence=True)  # (B, L, H)
            pred = decoder(z)  # (B, C, T)
            loss = mae_loss(pred, xb, mask)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # Validation
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
                loss = mae_loss(pred, xb, mask)
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
    """Load pre-trained weights into the encoder."""
    encoder.load_state_dict(state_dict)
