"""Pelatihan CNN-BiLSTM: loop, early stopping, dan determinisme seed.

Early stopping memonitor loss validasi dengan patience 8 mengikuti
Sebastian & Tantia (2024). Determinisme dicapai dengan menanam seed pada
seluruh sumber acak (torch, numpy, random) sehingga dua pelatihan dengan
seed sama menghasilkan loss identik; sebaran lintas seed dilaporkan
(Reimers & Gurevych 2017; Bouthillier et al. 2021).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lq45.models.cnn_bilstm import CNNBiLSTM
from lq45.models.encoder import CNNBiLSTMEncoder


def set_seed(seed: int) -> torch.Generator:
    """Tentukan seed semua sumber acak; kembalikan generator torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    """Peta `auto` ke CUDA bila tersedia, selain itu CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model(mk: dict[str, Any], seed: int) -> nn.Module:
    """Bangun CNN-BiLSTM dengan seed ditanam sebelum inisialisasi bobot.

    Inisialisasi memakai keadaan RNG global; tanpa seed di sini, dua
    panggilan berurutan menghasilkan bobot awal berbeda meski pelatihan
    kemudian disamakan seed-nya.
    """
    set_seed(seed)
    return CNNBiLSTM(**mk)


@dataclass
class FitResult:
    """Hasil satu pelatihan: bobot terbaik dan riwayat loss per epoch."""

    seed: int
    best_val_loss: float
    best_epoch: int
    stopped_early: bool
    state_dict: dict[str, Any]
    history: list[dict[str, float]]


def _predict_tensor(
    model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Prediksi dalam bentuk tensor; dipakai di dalam loop pelatihan."""
    potongan: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            potongan.append(model(xb))
    return torch.cat(potongan)


def predict(
    model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    """Prediksi untuk larik jendela `(N, F, w)`; hasil numpy float32."""
    model.eval()
    return _predict_tensor(model, x, batch_size, device).cpu().numpy()


def train_model(
    model: nn.Module,
    train_xy: tuple[np.ndarray, np.ndarray],
    val_xy: tuple[np.ndarray, np.ndarray],
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> FitResult:
    """Latih `model` dengan early stopping pada loss validasi.

    Bobot dengan loss validasi terbaik dipulihkan sebelum fungsi kembali.
    Model dibangun lewat `build_model` agar bobot awal ikut deterministik;
    fungsi ini tetap menanam seed sebelum loop pelatihan.
    """
    x_latih, y_latih = train_xy
    x_val, y_val = val_xy
    if len(y_latih) == 0 or len(y_val) == 0:
        raise ValueError(
            f"set pelatihan kosong: latih {len(y_latih)} baris, "
            f"validasi {len(y_val)} baris"
        )

    generator = set_seed(seed)
    model.to(device)
    dataset = TensorDataset(torch.from_numpy(x_latih), torch.from_numpy(y_latih))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    # Adam dan learning rate: Chaweewanchon & Chaysiri (2022) Bagian 4.1.3;
    # Sebastian & Tantia (2024).
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # MSE: Chaweewanchon & Chaysiri (2022); Kim et al. (2025).
    loss_fn = nn.MSELoss()
    y_val_tensor = torch.from_numpy(y_val).to(device)

    terbaik = float("inf")
    epoch_terbaik = 0
    state_terbaik: dict[str, Any] = {}
    menunggu = 0
    riwayat: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        jumlah = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            rugi = loss_fn(model(xb), yb)
            rugi.backward()
            optimizer.step()
            total += rugi.item() * len(yb)
            jumlah += len(yb)
        train_loss = total / max(jumlah, 1)

        model.eval()
        with torch.no_grad():
            pred = _predict_tensor(model, x_val, batch_size, device)
            val_loss = float(loss_fn(pred, y_val_tensor).item())
        riwayat.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < terbaik:
            terbaik = val_loss
            epoch_terbaik = epoch
            state_terbaik = {
                kunci: nilai.clone() for kunci, nilai in model.state_dict().items()
            }
            menunggu = 0
        else:
            menunggu += 1
            if menunggu >= patience:
                break

    model.load_state_dict(state_terbaik)
    return FitResult(
        seed=seed,
        best_val_loss=terbaik,
        best_epoch=epoch_terbaik,
        stopped_early=epoch < epochs,
        state_dict=state_terbaik,
        history=riwayat,
    )


def fine_tune_model(
    model: CNNBiLSTM,
    train_xy: tuple[np.ndarray, np.ndarray],
    val_xy: tuple[np.ndarray, np.ndarray],
    batch_size: int,
    epochs: int,
    patience: int,
    head_lr: float,
    encoder_lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    freeze_encoder: bool = False,
) -> FitResult:
    """Fine-tune model dengan discriminative learning rates.

    Args:
        model: CNNBiLSTM dengan encoder pre-trained
        train_xy, val_xy: data (X, y)
        batch_size, epochs, patience: training config
        head_lr: learning rate untuk projection head (dense)
        encoder_lr: learning rate untuk encoder (biasanya 10x lebih kecil)
        weight_decay: weight decay
        seed: random seed
        device: torch device
        freeze_encoder: jika True, hanya head yang dilatih

    Returns:
        FitResult dengan best state
    """
    x_latih, y_latih = train_xy
    x_val, y_val = val_xy
    if len(y_latih) == 0 or len(y_val) == 0:
        raise ValueError(
            f"set pelatihan kosong: latih {len(y_latih)} baris, "
            f"validasi {len(y_val)} baris"
        )

    generator = set_seed(seed)
    model.to(device)

    # Parameter groups dengan LR berbeda
    if freeze_encoder:
        model.freeze_encoder()
        params = [{"params": model.dense.parameters(), "lr": head_lr}]
    else:
        model.unfreeze_encoder()
        params = [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.dense.parameters(), "lr": head_lr},
        ]

    dataset = TensorDataset(torch.from_numpy(x_latih), torch.from_numpy(y_latih))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    optimizer = torch.optim.Adam(params, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    y_val_tensor = torch.from_numpy(y_val).to(device)

    terbaik = float("inf")
    epoch_terbaik = 0
    state_terbaik: dict[str, Any] = {}
    menunggu = 0
    riwayat: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        jumlah = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            rugi = loss_fn(model(xb), yb)
            rugi.backward()
            optimizer.step()
            total += rugi.item() * len(yb)
            jumlah += len(yb)
        train_loss = total / max(jumlah, 1)

        model.eval()
        with torch.no_grad():
            pred = _predict_tensor(model, x_val, batch_size, device)
            val_loss = float(loss_fn(pred, y_val_tensor).item())
        riwayat.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < terbaik:
            terbaik = val_loss
            epoch_terbaik = epoch
            state_terbaik = {
                kunci: nilai.clone() for kunci, nilai in model.state_dict().items()
            }
            menunggu = 0
        else:
            menunggu += 1
            if menunggu >= patience:
                break

    model.load_state_dict(state_terbaik)
    return FitResult(
        seed=seed,
        best_val_loss=terbaik,
        best_epoch=epoch_terbaik,
        stopped_early=epoch < epochs,
        state_dict=state_terbaik,
        history=riwayat,
    )
