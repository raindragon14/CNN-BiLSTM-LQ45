#!/usr/bin/env python3
"""Tahap 3: latih CNN-BiLSTM dan tulis prediksi walk-forward.

Mode:
- `dry-run`           : cetak tabel lipatan tanpa melatih.
- `smoke`             : 2 lipatan, 1 seed, 3 epoch (verifikasi pipa).
- `calibrate`         : ukur detik per epoch pada lipatan terkecil dan terbesar.
- `tune`              : grid dan sensitivitas periode desain, lalu bekukan
                         hyperparameter terbaik ke `best_hyperparams.yaml`.
- `pretrain`          : Masked Autoencoder pre-training pada data OHLCV+makro.
- `walk-forward`      : prediksi OOS 2020-2025 untuk tahap 4 (supervised only).
- `walk-forward-pretrain`: pre-train -> fine-tune per lipatan (two-stage).

Keluaran per run (`experiments/<run_id>/`):
- `run_info.json`      : cuplikan konfigurasi, sha git, versi pustaka.
- `fold_metrics.csv`   : loss, epoch terbaik, detik per epoch per pelatihan.
- `predictions.csv`    : satu baris per (tanggal, saham, seed); kolom
                          `role` menandai test atau validation.
- `predictions_parts/` : bagian mentah sebelum penghapusan duplikat
                          (untuk audit).
- `checkpoints/`       : state_dict per (lipatan, seed), opsional.
- `log.txt`            : kemajuan dengan stempel waktu.
- `pretrained/`        : encoder/decoder pre-trained (mode pretrain).

Aturan penghapusan duplikat prediksi: jendela test menutup seluruh OOS
tanpa tumpang tindih, sedangkan jendela validasi lipatan berikutnya
tumpang tindih dengan test lipatan sebelumnya (langkah 21 < test 21).
Setiap (tanggal, saham, seed) dipertahankan dari lipatan terkecil yang
memuat tanggal itu, sehingga tepat satu baris per (tanggal, saham, seed);
peran terekam di kolom `role` dan evaluasi tahap 5 menyaring `role=test`.

Spesifikasi: configs/model.yaml, configs/split.yaml, configs/universe.yaml.
Jejak keputusan: docs/keputusan_desain.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import yaml

from lq45.features.preprocess import RobustPreprocessor
from lq45.models.dataset import (
    FEATURE_COLUMNS,
    PRETRAIN_CHANNELS,
    PanelData,
    build_pretrain_windows,
    build_windows,
    forward_log_return,
    load_panel,
    load_raw_stocks,
)
from lq45.models.decoder import MAEDecoder
from lq45.models.encoder import CNNBiLSTMEncoder
from lq45.models.pretrain import (
    load_pretrained_encoder,
    pretrain_mae,
)
from lq45.models.training import (
    build_model,
    fine_tune_model,
    predict,
    resolve_device,
    set_seed,
    train_model,
)
from lq45.models.walkforward import Fold, design_split, make_folds, make_pretrain_split
from lq45.utils.config import (
    EXPERIMENT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_dirs,
    load_config,
)

SEEDS_BAWAAN = "0,1,2,3,4"


def parse_args() -> argparse.Namespace:
    """Baca argumen baris perintah."""
    parser = argparse.ArgumentParser(
        description="Latih CNN-BiLSTM dan tulis prediksi walk-forward."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "walk-forward",
            "walk-forward-pretrain",
            "pretrain",
            "tune",
            "smoke",
            "calibrate",
            "dry-run",
        ],
        default="walk-forward",
        help="mode kerja (bawaan: walk-forward)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=f"daftar seed dipisahkan dengan koma (bawaan: {SEEDS_BAWAAN})",
    )
    parser.add_argument("--jobs", type=int, default=1, help="jumlah proses paralel")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="thread torch per proses (0 = otomatis)",
    )
    parser.add_argument(
        "--folds", type=int, default=0, help="batasi lipatan (0 = semua)"
    )
    parser.add_argument(
        "--epochs", type=int, default=0, help="batasi epoch (0 = config)"
    )
    parser.add_argument(
        "--use-tuning",
        type=Path,
        default=None,
        help="direktori hasil tuning untuk membekukan hyperparameter",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="tanpa menyimpan state_dict",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="lewati (lipatan, seed) yang selesai untuk melanjutkan "
        "eksekusi terputus",
    )
    parser.add_argument("--out", type=Path, default=None, help="direktori keluaran")
    return parser.parse_args()


def seed_list(teks: str | None) -> list[int]:
    """Ubah daftar seed teks menjadi bilangan."""
    return [int(x) for x in (teks or SEEDS_BAWAAN).split(",") if x.strip()]


def nama_run(mode: str) -> Path:
    """Nama direktori hasil dari stempel waktu UTC."""
    stempel = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return EXPERIMENT_DIR / f"{mode}_{stempel}"


def git_sha() -> str:
    """Hash pendek commit terakhir; `unknown` bila bukan repositori git."""
    try:
        hasil = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return hasil.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log(run_dir: Path, pesan: str) -> None:
    """Tulis satu baris log ke layar dan `log.txt`."""
    baris = f"[{datetime.now(UTC).isoformat()}] {pesan}"
    print(baris, flush=True)
    with (run_dir / "log.txt").open("a", encoding="utf-8") as pegangan:
        pegangan.write(baris + "\n")


def tulis_json(path: Path, data: dict[str, Any]) -> None:
    """Tulis JSON secara atomik (berkas sementara lalu rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sementara = path.with_suffix(".tmp")
    with sementara.open("w", encoding="utf-8") as pegangan:
        json.dump(data, pegangan, indent=2, default=str)
    sementara.replace(path)


def muat_tuning(path: Path | None) -> dict[str, Any]:
    """Baca `best_hyperparams.yaml` dari hasil tuning; kosong bila tak ada."""
    if path is None:
        return {}
    berkas = path / "best_hyperparams.yaml"
    if not berkas.exists():
        raise FileNotFoundError(f"berkas tuning tidak ditemukan: {berkas}")
    return yaml.safe_load(berkas.read_text(encoding="utf-8"))


def model_kwargs(model_cfg: dict[str, Any], hp: dict[str, Any]) -> dict:
    """Susun parameter konstruktor CNNBiLSTM dari config.

    Kunci yang ada di `hp` menggantikan nilai config; `hp` berasal dari
    `best_hyperparams.yaml` hasil penyetelan periode desain.
    """
    return {
        "n_features": len(FEATURE_COLUMNS),
        "filters": list(model_cfg["cnn"]["filters"]),
        "kernel_size": int(model_cfg["cnn"]["kernel_size"]),
        "pooling": int(hp.get("pooling.size", model_cfg["cnn"]["pooling"]["size"])),
        "units": int(hp.get("bilstm.units", model_cfg["bilstm"]["units"])),
        "layers": int(model_cfg["bilstm"]["layers"]),
        "batchnorm": bool(model_cfg["cnn"]["batchnorm"]),
        "dropout_cnn": float(model_cfg["dropout"]["cnn"]),
        "dropout_lstm": float(model_cfg["dropout"]["lstm"]),
        "dropout_dense": float(model_cfg["dropout"]["dense"]),
        "bidirectional": bool(model_cfg["bilstm"]["bidirectional"]),
        "activation": model_cfg["cnn"]["activation"],
    }


def train_kwargs(
    model_cfg: dict[str, Any],
    hp: dict[str, Any],
    epochs: int = 0,
) -> dict:
    """Susun parameter pelatihan dari config; `hp` menggantikan nilainya."""
    return {
        "batch_size": int(hp.get("batch_size", model_cfg["batch_size"])),
        "epochs": epochs or int(model_cfg["epochs"]),
        "patience": int(model_cfg["early_stopping"]["patience"]),
        "lr": float(model_cfg["optimizer"]["lr"]),
        "weight_decay": float(
            hp.get(
                "optimizer.weight_decay",
                model_cfg["optimizer"]["weight_decay"],
            )
        ),
    }


def skala_panel(panel: PanelData, rentang_latih: tuple[int, int]) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    RobustPreprocessor,
    RobustPreprocessor,
]:
    """Winsorize + min-max di-fit pada baris latih, dipakai ke seluruh panel.

    Praproses mengikuti Sebastian & Tantia (2024) bagian praproses data.
    Parameter dihitung dari baris tanggal `rentang_latih` seluruh saham
    digabung, selaras dengan model bersama; transformasi bekerja per
    elemen, sehingga menerapkannya sebelum pembentukan jendela sama
    hasilnya dengan menerapkannya pada tiap jendela. Fit hanya pada data
    latih mencegah kebocoran informasi.
    """
    a, b = rentang_latih
    latih_fitur = np.concatenate(
        [panel.features[t][a:b] for t in panel.tickers], axis=0
    )
    latih_target = np.concatenate([panel.target[t][a:b] for t in panel.tickers], axis=0)
    prep_fitur = RobustPreprocessor().fit(
        pd.DataFrame(latih_fitur, columns=FEATURE_COLUMNS)
    )
    prep_target = RobustPreprocessor().fit(pd.DataFrame({"target": latih_target}))

    fitur = {
        t: prep_fitur.transform(
            pd.DataFrame(panel.features[t], columns=FEATURE_COLUMNS)
        ).to_numpy(dtype=np.float32)
        for t in panel.tickers
    }
    target = {
        t: prep_target.transform(pd.DataFrame({"target": panel.target[t]}))[
            "target"
        ].to_numpy(dtype=np.float32)
        for t in panel.tickers
    }
    return fitur, target, prep_fitur, prep_target


def jendela_pool(
    panel: PanelData,
    fitur: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    rentang: tuple[int, int],
    banned: list[tuple[int, int]],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gabungkan jendela seluruh saham untuk rentang tanggal tertentu."""
    daftar_x: list[np.ndarray] = []
    daftar_y: list[np.ndarray] = []
    for ticker in panel.tickers:
        x, y, _ = build_windows(
            fitur[ticker],
            target[ticker],
            lookback,
            rentang[0],
            rentang[1],
            banned,
        )
        if len(y):
            daftar_x.append(x)
            daftar_y.append(y)
    if not daftar_x:
        kosong = np.empty((0, len(FEATURE_COLUMNS), lookback), dtype=np.float32)
        return kosong, np.empty(0, dtype=np.float32)
    return np.concatenate(daftar_x), np.concatenate(daftar_y)


def jalankan_fit(
    panel: PanelData,
    fold: Fold,
    lookback: int,
    mk: dict[str, Any],
    tk: dict[str, Any],
    seed: int,
    device: torch.device,
    run_dir: Path,
    simpan_checkpoint: bool,
) -> dict[str, Any]:
    """Latih satu (lipatan, seed) lalu tulis bagian prediksi dan metrik."""
    mulai = time.time()
    fitur, target, _, prep_target = skala_panel(panel, fold.train)
    x_train, y_train = jendela_pool(
        panel, fitur, target, fold.train, fold.banned, lookback
    )
    x_val, y_val = jendela_pool(panel, fitur, target, fold.validation, [], lookback)
    x_test, y_test = jendela_pool(panel, fitur, target, fold.test, [], lookback)

    model = build_model(mk, seed)
    hasil = train_model(
        model,
        (x_train, y_train),
        (x_val, y_val),
        tk["batch_size"],
        tk["epochs"],
        tk["patience"],
        tk["lr"],
        tk["weight_decay"],
        seed,
        device,
    )

    pred_test = predict(model, x_test, tk["batch_size"], device)
    test_loss = float(np.mean((pred_test - y_test) ** 2))

    baris: list[dict[str, Any]] = []
    for rentang, peran in (
        (fold.validation, "validation"),
        (fold.test, "test"),
    ):
        for ticker in panel.tickers:
            x, y, idx = build_windows(
                fitur[ticker],
                target[ticker],
                lookback,
                rentang[0],
                rentang[1],
                [],
            )
            if len(y) == 0:
                continue
            pred = predict(model, x, tk["batch_size"], device)
            pred_raw = prep_target.inverse_transform(pd.DataFrame({"target": pred}))[
                "target"
            ].to_numpy()
            y_raw = prep_target.inverse_transform(pd.DataFrame({"target": y}))[
                "target"
            ].to_numpy()
            for j, posisi in enumerate(idx):
                baris.append(
                    {
                        "fold": fold.id,
                        "seed": seed,
                        "date": panel.dates[posisi].date().isoformat(),
                        "ticker": ticker,
                        "role": peran,
                        "pred_scaled": float(pred[j]),
                        "true_scaled": float(y[j]),
                        "pred_raw": float(pred_raw[j]),
                        "true_raw": float(y_raw[j]),
                    }
                )

    bagian = pd.DataFrame(
        baris,
        columns=[
            "fold",
            "seed",
            "date",
            "ticker",
            "role",
            "pred_scaled",
            "true_scaled",
            "pred_raw",
            "true_raw",
        ],
    )
    bagian.to_csv(
        run_dir / "predictions_parts" / f"{fold.id:03d}_{seed}.csv",
        index=False,
    )
    if simpan_checkpoint:
        torch.save(
            hasil.state_dict,
            run_dir / "checkpoints" / f"{fold.id:03d}_{seed}.pt",
        )

    metrik = {
        "fold": fold.id,
        "seed": seed,
        "train_loss": hasil.history[-1]["train_loss"],
        "val_loss": hasil.best_val_loss,
        "test_loss": test_loss,
        "best_epoch": hasil.best_epoch,
        "stopped_early": hasil.stopped_early,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "wall_sec": round(time.time() - mulai, 3),
        "sec_per_epoch": round((time.time() - mulai) / max(len(hasil.history), 1), 3),
    }
    tulis_json(run_dir / "metrics" / f"{fold.id:03d}_{seed}.json", metrik)
    return metrik


def latih_desain(
    panel: PanelData,
    design: Any,
    lookback: int,
    mk: dict[str, Any],
    tk: dict[str, Any],
    seed: int,
    device: torch.device,
) -> float:
    """Latih pada pembagian periode desain; kembalikan loss validasi.

    Purge sudah melekat pada `design.train`; target sudah dibangun
    sebelumnya oleh pemanggil sesuai horizon yang diuji.
    """
    fitur, target, _, _ = skala_panel(panel, design.train)
    x_train, y_train = jendela_pool(
        panel, fitur, target, design.train, design.banned, lookback
    )
    x_val, y_val = jendela_pool(panel, fitur, target, design.validation, [], lookback)
    model = build_model(mk, seed)
    hasil = train_model(
        model,
        (x_train, y_train),
        (x_val, y_val),
        tk["batch_size"],
        tk["epochs"],
        tk["patience"],
        tk["lr"],
        tk["weight_decay"],
        seed,
        device,
    )
    return hasil.best_val_loss


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Kembangkan kisi bertingkat menjadi daftar kombinasi."""
    kombinasi: list[dict[str, Any]] = [{}]
    for kunci, nilai in grid.items():
        kombinasi = [{**k, kunci: v} for k in kombinasi for v in nilai]
    return kombinasi


def sudah_selesai(run_dir: Path, id_lipatan: int, seed: int) -> bool:
    """True bila bagian prediksi dan metrik pasangan ini sudah ada."""
    bagian = run_dir / "predictions_parts" / f"{id_lipatan:03d}_{seed}.csv"
    metrik = run_dir / "metrics" / f"{id_lipatan:03d}_{seed}.json"
    return bagian.exists() and metrik.exists()


def muat_panel(tickers: list[str], horizon: int) -> PanelData:
    """Muat panel; saham tanpa berkas terproses dilewati dengan peringatan.

    Universe resmi berisi 45 konstituen, tetapi dua di antaranya (SRIL,
    WSKT) tidak tersedia di Yahoo Finance; daftar efektif dicatat di
    `run_info.json` lewat `panel.tickers`.
    """
    direktori = PROCESSED_DIR / "features"
    tersedia = [
        t for t in tickers if (direktori / f"{t.replace('.JK', '')}.csv").exists()
    ]
    hilang = [t for t in tickers if t not in tersedia]
    if hilang:
        print(
            f"peringatan: {len(hilang)} saham tanpa berkas terproses "
            f"dilewati: {hilang}"
        )
    if not tersedia:
        raise RuntimeError("tidak ada saham dengan berkas terproses")
    return load_panel(direktori, tersedia, horizon)


def gabungkan(run_dir: Path) -> pd.DataFrame:
    """Gabungkan bagian prediksi dan metrik menjadi keluaran akhir.

    Penghapusan duplikat: setiap (tanggal, saham, seed) dipertahankan satu
    baris dari lipatan terkecil yang memuat tanggal itu. Tanggal OOS
    pertama muncul di validasi lipatan 0, sisanya muncul di jendela test
    lipatan yang menaunginya; validasi lipatan berikutnya yang tumpang
    tindih dengan test terdahulu tersingkir dengan sendirinya.
    """
    berkas_metrik = sorted((run_dir / "metrics").glob("*.json"))
    if not berkas_metrik:
        raise RuntimeError(f"tidak ada metrik di {run_dir}")
    metrik = pd.DataFrame(
        [json.loads(p.read_text(encoding="utf-8")) for p in berkas_metrik]
    )
    metrik.sort_values(["fold", "seed"]).to_csv(
        run_dir / "fold_metrics.csv", index=False
    )

    berkas_bagian = sorted((run_dir / "predictions_parts").glob("*.csv"))
    bagian = pd.concat([pd.read_csv(p) for p in berkas_bagian], ignore_index=True)
    bagian = bagian.sort_values("fold")
    tanpa_duplikat = bagian.drop_duplicates(
        subset=["date", "ticker", "seed"], keep="first"
    )
    tanpa_duplikat = tanpa_duplikat.sort_values(["date", "ticker", "seed"]).reset_index(
        drop=True
    )
    tanpa_duplikat.to_csv(run_dir / "predictions.csv", index=False)
    return tanpa_duplikat


def ringkasan_smoke(run_dir: Path, prediksi: pd.DataFrame) -> None:
    """Pemeriksaan awal ringkas untuk mode smoke.

    Korelasi prediksi dengan aktual di pasar saham cenderung kecil;
    nilainya dicetak agar anomali (misalnya kebocoran data) langsung
    terlihat.
    """
    korelasi = prediksi[["pred_raw", "true_raw"]].corr().iloc[0, 1]
    mse_raw = float(np.mean((prediksi["pred_raw"] - prediksi["true_raw"]) ** 2))
    print(
        f"uji awal: korelasi prediksi dan aktual = {korelasi:.4f} "
        f"(nilai kecil adalah wajar), MSE ruang asal = {mse_raw:.6f}"
    )
    log(
        run_dir,
        f"uji awal: korelasi {korelasi:.4f}, MSE ruang asal {mse_raw:.6f}",
    )


def tulis_run_info(
    run_dir: Path,
    mode: str,
    seeds: list[int],
    jobs: int,
    threads: int,
    lookback: int,
    horizon: int,
    n_lipatan: int,
    device: str,
    use_tuning: Path | None,
) -> None:
    """Tulis rekaman eksekusi untuk audit."""
    info = {
        "mode": mode,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "seeds": seeds,
        "jobs": jobs,
        "threads": threads,
        "lookback_days": lookback,
        "horizon_days": horizon,
        "n_folds": n_lipatan,
        "device": device,
        "use_tuning": str(use_tuning) if use_tuning else None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "configs": {
            nama: load_config(nama)
            for nama in ("data", "split", "model", "portfolio", "experiment")
        },
    }
    tulis_json(run_dir / "run_info.json", info)


def run_dryrun(args: argparse.Namespace) -> int:
    """Cetak tabel lipatan tanpa melatih."""
    model_cfg = load_config("model")
    split_cfg = load_config("split")
    tickers = list(load_config("universe")["tickers"])
    hp = muat_tuning(args.use_tuning)
    lookback = int(hp.get("lookback_days", model_cfg["lookback_days"]))
    horizon = int(hp.get("horizon_days", model_cfg["horizon_days"]))
    panel = muat_panel(tickers, horizon)
    lipatan = make_folds(
        len(panel.dates),
        split_cfg,
        purge_days=horizon,
        embargo_days=lookback - 1,
    )
    if args.folds:
        lipatan = lipatan[: args.folds]

    print(
        f"kalender: {panel.dates[0].date()} s.d. {panel.dates[-1].date()}"
        f" ({len(panel.dates)} hari), w={lookback}, tau={horizon}"
    )
    kepala = (
        f"{'id':>3} {'latih s.d.':>12} {'validasi':>23} {'test':>23} "
        f"{'n_latih':>9} {'n_val':>7} {'n_test':>7} {'larangan':>9}"
    )
    print(kepala)
    for fold in lipatan:
        n_latih = sum(
            len(
                build_windows(
                    panel.features[t],
                    panel.target[t],
                    lookback,
                    fold.train[0],
                    fold.train[1],
                    fold.banned,
                )[1]
            )
            for t in panel.tickers
        )
        n_val = sum(
            len(
                build_windows(
                    panel.features[t],
                    panel.target[t],
                    lookback,
                    fold.validation[0],
                    fold.validation[1],
                    [],
                )[1]
            )
            for t in panel.tickers
        )
        n_test = sum(
            len(
                build_windows(
                    panel.features[t],
                    panel.target[t],
                    lookback,
                    fold.test[0],
                    fold.test[1],
                    [],
                )[1]
            )
            for t in panel.tickers
        )
        rentang = lambda ab: (
            f"{panel.dates[ab[0]].date()} - {panel.dates[ab[1] - 1].date()}"
        )
        print(
            f"{fold.id:>3} {panel.dates[fold.train[1] - 1].date():>12} "
            f"{rentang(fold.validation):>23} {rentang(fold.test):>23} "
            f"{n_latih:>9} {n_val:>7} {n_test:>7} {len(fold.banned):>9}"
        )
    print(f"jumlah lipatan: {len(lipatan)}")
    return 0


def run_tune(args: argparse.Namespace) -> int:
    """Setel hyperparameter pada periode desain (grid + sensitivitas)."""
    model_cfg = load_config("model")
    tickers = list(load_config("universe")["tickers"])
    seeds = seed_list(args.seeds)
    device = resolve_device(model_cfg["device"])
    torch.set_num_threads(args.threads or torch.get_num_threads())

    run_dir = args.out or nama_run("tune")
    ensure_dirs(run_dir)
    log(run_dir, f"tuning dimulai: {len(seeds)} seed, perangkat {device}")

    panel = muat_panel(tickers, 5)
    tune_cfg = model_cfg["tuning"]
    design = design_split(
        panel.dates, tune_cfg["train"], tune_cfg["validate"], horizon=5
    )
    log(
        run_dir,
        f"periode desain: latih {panel.dates[design.train[0]].date()} s.d. "
        f"{panel.dates[design.train[1] - 1].date()}; validasi "
        f"{panel.dates[design.validation[0]].date()} s.d. "
        f"{panel.dates[design.validation[1] - 1].date()}",
    )

    grid = expand_grid(tune_cfg["grid"])
    baris: list[dict[str, Any]] = []
    for i, hp in enumerate(grid, 1):
        mk = model_kwargs(model_cfg, hp)
        tk = train_kwargs(model_cfg, hp)
        skor = [latih_desain(panel, design, 60, mk, tk, seed, device) for seed in seeds]
        baris.append(
            {
                **hp,
                "val_loss_mean": float(np.mean(skor)),
                "val_loss_std": float(np.std(skor)),
            }
        )
        log(
            run_dir,
            f"grid {i}/{len(grid)}: {hp} -> mean {np.mean(skor):.6f}",
        )
    grid_df = pd.DataFrame(baris).sort_values("val_loss_mean")
    grid_df.to_csv(run_dir / "grid_results.csv", index=False)
    terbaik_hp = {
        kunci: nilai
        for kunci, nilai in grid_df.iloc[0].items()
        if kunci in tune_cfg["grid"]
    }
    log(run_dir, f"grid terbaik: {terbaik_hp}")

    mk = model_kwargs(model_cfg, terbaik_hp)
    tk = train_kwargs(model_cfg, terbaik_hp)
    baris_sens: list[dict[str, Any]] = []
    sens_cfg = model_cfg.get("sensitivity", {})
    for w in sens_cfg.get("lookback_days", [60]):
        for tau in sens_cfg.get("horizon_days", [5]):
            target = {t: forward_log_return(panel.close[t], tau) for t in panel.tickers}
            panel_tau = dataclasses.replace(panel, target=target)
            design_tau = design_split(
                panel.dates, tune_cfg["train"], tune_cfg["validate"], tau
            )
            skor = [
                latih_desain(panel_tau, design_tau, w, mk, tk, seed, device)
                for seed in seeds
            ]
            baris_sens.append(
                {
                    "lookback_days": w,
                    "horizon_days": tau,
                    "val_loss_mean": float(np.mean(skor)),
                    "val_loss_std": float(np.std(skor)),
                }
            )
            log(
                run_dir,
                f"sensitivitas w={w} tau={tau} -> " f"mean {np.mean(skor):.6f}",
            )
    sens_df = pd.DataFrame(baris_sens).sort_values("val_loss_mean")
    sens_df.to_csv(run_dir / "sensitivity_results.csv", index=False)

    terbaik = dict(grid_df.iloc[0])
    terbaik_w_tau = dict(sens_df.iloc[0])
    best = {
        "bilstm.units": int(terbaik["bilstm.units"]),
        "batch_size": int(terbaik["batch_size"]),
        "optimizer.weight_decay": float(terbaik["optimizer.weight_decay"]),
        "pooling.size": int(terbaik["pooling.size"]),
        "lookback_days": int(terbaik_w_tau["lookback_days"]),
        "horizon_days": int(terbaik_w_tau["horizon_days"]),
        "grid_val_loss_mean": float(terbaik["val_loss_mean"]),
        "sensitivity_val_loss_mean": float(terbaik_w_tau["val_loss_mean"]),
        "seeds": seeds,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
    }
    with (run_dir / "best_hyperparams.yaml").open("w", encoding="utf-8") as pegangan:
        yaml.safe_dump(best, pegangan, sort_keys=False, allow_unicode=True)
    log(run_dir, f"hyperparameter terbaik dibekukan: {best}")
    return 0


def run_calibrate(args: argparse.Namespace) -> int:
    """Ukur waktu pelatihan pada lipatan terkecil dan terbesar."""
    model_cfg = load_config("model")
    split_cfg = load_config("split")
    tickers = list(load_config("universe")["tickers"])
    hp = muat_tuning(args.use_tuning)
    lookback = int(hp.get("lookback_days", model_cfg["lookback_days"]))
    horizon = int(hp.get("horizon_days", model_cfg["horizon_days"]))
    mk = model_kwargs(model_cfg, hp)
    tk = train_kwargs(model_cfg, hp, args.epochs)
    device = resolve_device(model_cfg["device"])
    torch.set_num_threads(args.threads or torch.get_num_threads())

    panel = muat_panel(tickers, horizon)
    lipatan = make_folds(
        len(panel.dates),
        split_cfg,
        purge_days=horizon,
        embargo_days=lookback - 1,
    )
    run_dir = args.out or nama_run("calibrate")
    ensure_dirs(
        run_dir / "predictions_parts",
        run_dir / "checkpoints",
        run_dir / "metrics",
    )
    log(run_dir, f"kalibrasi dimulai: {len(lipatan)} lipatan tersedia")

    ukuran: list[dict[str, Any]] = []
    for fold in (lipatan[0], lipatan[-1]):
        metrik = jalankan_fit(panel, fold, lookback, mk, tk, 0, device, run_dir, False)
        ukuran.append(metrik)
        log(
            run_dir,
            f"lipatan {fold.id}: waktu {metrik['wall_sec']} detik, "
            f"{metrik['sec_per_epoch']} detik/epoch, "
            f"{metrik['best_epoch']} epoch",
        )
    pd.DataFrame(ukuran).to_csv(run_dir / "calibrate.csv", index=False)

    n_seed = len(seed_list(args.seeds))
    n_fit = len(lipatan) * n_seed
    cepat = ukuran[0]["wall_sec"] * n_fit
    lambat = ukuran[1]["wall_sec"] * n_fit
    log(
        run_dir,
        f"ekstrapolasi {n_fit} pelatihan: "
        f"{cepat / 3600:.1f} jam (laju lipatan awal) s.d. "
        f"{lambat / 3600:.1f} jam (laju lipatan akhir)",
    )
    print(
        f"perkiraan waktu total: {cepat / 3600:.1f} - "
        f"{lambat / 3600:.1f} jam untuk {n_fit} pelatihan "
        f"({len(lipatan)} lipatan x {n_seed} seed)"
    )
    return 0


def run_pretrain(args: argparse.Namespace) -> int:
    """Masked Autoencoder pre-training pada data OHLCV + makro (7 channel).

    Pre-training self-supervised tanpa label return. Encoder belajar
    representasi harga universal dari semua saham LQ45.
    """
    model_cfg = load_config("model")
    pretrain_cfg = model_cfg.get("pretrain", {})

    tickers = list(load_config("universe")["tickers"])
    seeds = seed_list(args.seeds)
    device = resolve_device(model_cfg["device"])
    torch.set_num_threads(args.threads or torch.get_num_threads())

    run_dir = args.out or nama_run("pretrain")
    ensure_dirs(
        run_dir / "pretrained",
        run_dir / "metrics",
    )
    log(run_dir, f"pre-training MAE dimulai: {len(seeds)} seed, perangkat {device}")

    if not pretrain_cfg.get("enabled", False):
        log(run_dir, "pre-training disabled in config (pretrain.enabled=false)")
        return 1

    # Load raw stocks (OHLCV + BI-7DRRR + JISDOR = 7 channel) per saham
    stocks = load_raw_stocks(RAW_DIR, tickers)
    log(run_dir, f"loaded {len(stocks)} saham untuk pre-training")

    # Build pretrain windows from all stocks
    lookback = int(model_cfg["lookback_days"])

    # First pass: compute splits and collect training data for preprocessing
    stock_splits = []
    all_train_data = []

    for stock in stocks:
        stock_len = len(stock.features)
        pt_split = make_pretrain_split(stock_len, pretrain_cfg.get("val_ratio", 0.1))
        stock_splits.append((stock, pt_split))
        # Collect training portion for fitting preprocessor
        train_feat = stock.features[pt_split.train[0] : pt_split.train[1]]
        if len(train_feat) > 0:
            all_train_data.append(train_feat)

    # Fit RobustPreprocessor on combined training data (winsorize + min-max)
    # This normalizes the 7 channels (OHLCV + BI-7DRRR + JISDOR) to prevent
    # scale issues like JISDOR ~16000 dominating the loss
    if all_train_data:
        combined_train = np.concatenate(all_train_data, axis=0)
        prep = RobustPreprocessor().fit(
            pd.DataFrame(combined_train, columns=PRETRAIN_CHANNELS)
        )
        log(
            run_dir,
            f"RobustPreprocessor fitted on {len(combined_train)} training samples",
        )
    else:
        log(
            run_dir,
            "WARNING: no training data for preprocessor, skipping normalization",
        )
        prep = None

    # Second pass: transform features and build windows
    all_train_x = []
    all_val_x = []

    for stock, pt_split in stock_splits:
        # Apply preprocessing if fitted
        if prep is not None:
            train_feat = prep.transform(
                pd.DataFrame(
                    stock.features[pt_split.train[0] : pt_split.train[1]],
                    columns=PRETRAIN_CHANNELS,
                )
            ).to_numpy(dtype=np.float32)
            val_feat = prep.transform(
                pd.DataFrame(
                    stock.features[pt_split.validation[0] : pt_split.validation[1]],
                    columns=PRETRAIN_CHANNELS,
                )
            ).to_numpy(dtype=np.float32)
        else:
            train_feat = stock.features[pt_split.train[0] : pt_split.train[1]]
            val_feat = stock.features[pt_split.validation[0] : pt_split.validation[1]]

        tx, _ = build_pretrain_windows(train_feat, lookback, 0, len(train_feat))
        vx, _ = build_pretrain_windows(val_feat, lookback, 0, len(val_feat))
        if len(tx):
            all_train_x.append(tx)
        if len(vx):
            all_val_x.append(vx)

    train_x = (
        np.concatenate(all_train_x, axis=0)
        if all_train_x
        else np.empty((0, 7, lookback), dtype=np.float32)
    )
    val_x = (
        np.concatenate(all_val_x, axis=0)
        if all_val_x
        else np.empty((0, 7, lookback), dtype=np.float32)
    )

    log(run_dir, f"pre-train samples: train {len(train_x)}, val {len(val_x)}")

    if len(train_x) == 0:
        log(run_dir, "tidak ada data pre-train")
        return 1

    # Run pre-training for each seed
    for seed in seeds:
        log(run_dir, f"pre-train seed {seed}...")
        set_seed(seed)

        encoder = CNNBiLSTMEncoder(
            n_features=7,
            filters=list(model_cfg["cnn"]["filters"]),
            kernel_size=int(model_cfg["cnn"]["kernel_size"]),
            pooling=int(model_cfg["cnn"]["pooling"]["size"]),
            units=int(model_cfg["bilstm"]["units"]),
            layers=int(model_cfg["bilstm"]["layers"]),
            batchnorm=bool(model_cfg["cnn"]["batchnorm"]),
            dropout_cnn=float(model_cfg["dropout"]["cnn"]),
            dropout_lstm=float(model_cfg["dropout"]["lstm"]),
            bidirectional=bool(model_cfg["bilstm"]["bidirectional"]),
            activation=model_cfg["cnn"]["activation"],
        )
        decoder = MAEDecoder(
            latent_dim=encoder.output_dim,
            n_channels=5,  # reconstruct OHLCV only
            lookback=lookback,
            pooling=int(model_cfg["cnn"]["pooling"]["size"]),
            hidden=pretrain_cfg.get("decoder", {}).get("hidden", 128),
            layers=pretrain_cfg.get("decoder", {}).get("layers", 2),
        )

        result = pretrain_mae(
            encoder=encoder,
            decoder=decoder,
            train_x=train_x,
            val_x=val_x,
            mask_ratio=pretrain_cfg.get("mask_ratio", 0.3),
            epochs=args.epochs or pretrain_cfg.get("epochs", 50),
            lr=pretrain_cfg.get("lr", 1e-3),
            weight_decay=pretrain_cfg.get("weight_decay", 0.0),
            batch_size=pretrain_cfg.get("batch_size", 64),
            patience=pretrain_cfg.get("patience", 10),
            device=device,
            seed=seed,
        )

        # Save pretrained encoder/decoder
        torch.save(
            result.encoder_state, run_dir / "pretrained" / f"encoder_seed{seed}.pt"
        )
        torch.save(
            result.decoder_state, run_dir / "pretrained" / f"decoder_seed{seed}.pt"
        )

        # Save metrics
        metrik = {
            "seed": seed,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "wall_sec": result.wall_sec,
        }
        tulis_json(run_dir / "metrics" / f"pretrain_seed{seed}.json", metrik)
        log(
            run_dir,
            f"seed {seed}: val_loss {result.best_val_loss:.6f}, epoch {result.best_epoch}",
        )

    # Save combined best (seed 0 as reference)
    tulis_json(
        run_dir / "metrics" / "pretrain_summary.json",
        {
            "seeds": seeds,
            "lookback": lookback,
            "pretrain_channels": list(PRETRAIN_CHANNELS),
            "config": pretrain_cfg,
        },
    )
    log(run_dir, f"pre-training selesai -> {run_dir / 'pretrained'}")
    return 0


def run_walkforward_pretrain(args: argparse.Namespace) -> int:
    """Walk-forward dengan pre-training: pre-train sekali, lalu fine-tune per lipatan.

    Alur:
    1. Jalankan pre-training (jika belum ada) untuk semua seed
    2. Per lipatan: load pretrained encoder -> fine-tune -> prediksi
    """
    model_cfg = load_config("model")
    pretrain_cfg = model_cfg.get("pretrain", {})
    split_cfg = load_config("split")
    tickers = list(load_config("universe")["tickers"])

    if args.mode == "smoke":
        seeds = seed_list(args.seeds)[:1]
        batas_lipatan = args.folds or 2
        batas_epoch = args.epochs or 3
    else:
        seeds = seed_list(args.seeds)
        batas_lipatan = args.folds
        batas_epoch = args.epochs

    hp = muat_tuning(args.use_tuning)
    lookback = int(hp.get("lookback_days", model_cfg["lookback_days"]))
    horizon = int(hp.get("horizon_days", model_cfg["horizon_days"]))
    mk = model_kwargs(model_cfg, hp)
    tk = train_kwargs(model_cfg, hp, batas_epoch)

    device = resolve_device(model_cfg["device"])
    threads = args.threads or torch.get_num_threads()
    if args.jobs > 1 and not args.threads:
        threads = max(1, min(4, (os.cpu_count() or 1) // args.jobs))
    torch.set_num_threads(threads)

    # Load panel dengan target (supervised)
    panel = muat_panel(tickers, horizon)
    lipatan = make_folds(
        len(panel.dates),
        split_cfg,
        purge_days=horizon,
        embargo_days=lookback - 1,
    )
    if batas_lipatan:
        lipatan = lipatan[:batas_lipatan]

    run_dir = args.out or nama_run("walk-forward-pretrain")
    ensure_dirs(
        run_dir / "predictions_parts",
        run_dir / "checkpoints",
        run_dir / "metrics",
        run_dir / "pretrained",
    )
    log(
        run_dir,
        f"mulai walk-forward-pretrain: {len(lipatan)} lipatan, {len(seeds)} seed, "
        f"jobs {args.jobs}, thread {threads}, perangkat {device}",
    )

    # Fine-tune config
    ft_cfg = pretrain_cfg.get("fine_tune", {})
    freeze_encoder = ft_cfg.get("freeze_encoder", False)
    head_lr = ft_cfg.get("head_lr", 1e-4)
    encoder_lr = ft_cfg.get("encoder_lr", 1e-5)
    ft_weight_decay = pretrain_cfg.get("weight_decay", 0.0)

    pasangan = [(fold.id, seed) for fold in lipatan for seed in seeds]
    pasangan = [
        p for p in pasangan if not (args.resume and sudah_selesai(run_dir, p[0], p[1]))
    ]

    if not pasangan:
        log(run_dir, "semua pasangan sudah selesai (mode lanjut)")
    elif args.jobs <= 1:
        for id_lipatan, seed in pasangan:
            fold = lipatan[id_lipatan]
            metrik = jalankan_fit_pretrain(
                panel,
                fold,
                lookback,
                mk,
                tk,
                seed,
                device,
                run_dir,
                not args.no_checkpoints,
                freeze_encoder,
                head_lr,
                encoder_lr,
                ft_weight_decay,
            )
            log(
                run_dir,
                f"lipatan {fold.id} seed {seed}: val "
                f"{metrik['val_loss']:.6f}, test {metrik['test_loss']:.6f}, "
                f"waktu {metrik['wall_sec']} detik",
            )
    else:
        # Parallel execution would need panel pickling - skip for now
        log(
            run_dir,
            "parallel jobs tidak didukung untuk walk-forward-pretrain (gunakan --jobs 1)",
        )
        for id_lipatan, seed in pasangan:
            fold = lipatan[id_lipatan]
            metrik = jalankan_fit_pretrain(
                panel,
                fold,
                lookback,
                mk,
                tk,
                seed,
                device,
                run_dir,
                not args.no_checkpoints,
                freeze_encoder,
                head_lr,
                encoder_lr,
                ft_weight_decay,
            )
            log(
                run_dir,
                f"lipatan {fold.id} seed {seed}: val "
                f"{metrik['val_loss']:.6f}, test {metrik['test_loss']:.6f}, "
                f"waktu {metrik['wall_sec']} detik",
            )

    prediksi = gabungkan(run_dir)
    tulis_run_info(
        run_dir,
        args.mode,
        seeds,
        args.jobs,
        threads,
        lookback,
        horizon,
        len(lipatan),
        str(device),
        args.use_tuning,
    )
    log(
        run_dir,
        f"selesai: {len(prediksi)} baris prediksi -> " f"{run_dir / 'predictions.csv'}",
    )
    if args.mode == "smoke":
        ringkasan_smoke(run_dir, prediksi)
    return 0


def jalankan_fit_pretrain(
    panel: PanelData,
    fold: Fold,
    lookback: int,
    mk: dict[str, Any],
    tk: dict[str, Any],
    seed: int,
    device: torch.device,
    run_dir: Path,
    simpan_checkpoint: bool,
    freeze_encoder: bool,
    head_lr: float,
    encoder_lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Latih satu (lipatan, seed) dengan pre-trained encoder + fine-tune."""
    mulai = time.time()

    # Load pretrained encoder for this seed
    pretrained_path = run_dir / "pretrained" / f"encoder_seed{seed}.pt"
    if not pretrained_path.exists():
        # Fallback: check for any pretrained encoder
        pretrained_files = list(run_dir.glob("pretrained/encoder_seed*.pt"))
        if pretrained_files:
            pretrained_path = pretrained_files[0]
            log(run_dir, f"fallback pretrained: {pretrained_path}")
        else:
            raise FileNotFoundError(
                f"pre-trained encoder tidak ditemukan: {pretrained_path}"
            )

    # Scale panel
    fitur, target, _, prep_target = skala_panel(panel, fold.train)
    x_train, y_train = jendela_pool(
        panel, fitur, target, fold.train, fold.banned, lookback
    )
    x_val, y_val = jendela_pool(panel, fitur, target, fold.validation, [], lookback)
    x_test, y_test = jendela_pool(panel, fitur, target, fold.test, [], lookback)

    # Build model with pretrained encoder
    model = build_model(mk, seed)
    load_pretrained_encoder(
        model.encoder, torch.load(pretrained_path, map_location=device)
    )

    # Fine-tune
    hasil = fine_tune_model(
        model,
        (x_train, y_train),
        (x_val, y_val),
        tk["batch_size"],
        tk["epochs"],
        tk["patience"],
        head_lr,
        encoder_lr,
        weight_decay,
        seed,
        device,
        freeze_encoder=freeze_encoder,
    )

    pred_test = predict(model, x_test, tk["batch_size"], device)
    test_loss = float(np.mean((pred_test - y_test) ** 2))

    baris: list[dict[str, Any]] = []
    for rentang, peran in (
        (fold.validation, "validation"),
        (fold.test, "test"),
    ):
        for ticker in panel.tickers:
            x, y, idx = build_windows(
                fitur[ticker],
                target[ticker],
                lookback,
                rentang[0],
                rentang[1],
                [],
            )
            if len(y) == 0:
                continue
            pred = predict(model, x, tk["batch_size"], device)
            pred_raw = prep_target.inverse_transform(pd.DataFrame({"target": pred}))[
                "target"
            ].to_numpy()
            y_raw = prep_target.inverse_transform(pd.DataFrame({"target": y}))[
                "target"
            ].to_numpy()
            for j, posisi in enumerate(idx):
                baris.append(
                    {
                        "fold": fold.id,
                        "seed": seed,
                        "date": panel.dates[posisi].date().isoformat(),
                        "ticker": ticker,
                        "role": peran,
                        "pred_scaled": float(pred[j]),
                        "true_scaled": float(y[j]),
                        "pred_raw": float(pred_raw[j]),
                        "true_raw": float(y_raw[j]),
                    }
                )

    bagian = pd.DataFrame(
        baris,
        columns=[
            "fold",
            "seed",
            "date",
            "ticker",
            "role",
            "pred_scaled",
            "true_scaled",
            "pred_raw",
            "true_raw",
        ],
    )
    bagian.to_csv(
        run_dir / "predictions_parts" / f"{fold.id:03d}_{seed}.csv",
        index=False,
    )
    if simpan_checkpoint:
        torch.save(
            hasil.state_dict,
            run_dir / "checkpoints" / f"{fold.id:03d}_{seed}.pt",
        )

    metrik = {
        "fold": fold.id,
        "seed": seed,
        "train_loss": hasil.history[-1]["train_loss"],
        "val_loss": hasil.best_val_loss,
        "test_loss": test_loss,
        "best_epoch": hasil.best_epoch,
        "stopped_early": hasil.stopped_early,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "wall_sec": round(time.time() - mulai, 3),
        "sec_per_epoch": round((time.time() - mulai) / max(len(hasil.history), 1), 3),
    }
    tulis_json(run_dir / "metrics" / f"{fold.id:03d}_{seed}.json", metrik)
    return metrik


@dataclass
class _KonteksPekerja:
    """Konteks proses pekerja; diisi oleh fungsi inisialisasi."""

    panel: PanelData | None = None
    lipatan: list[Fold] | None = None
    lookback: int | None = None
    mk: dict[str, Any] | None = None
    tk: dict[str, Any] | None = None
    device: torch.device | None = None
    run_dir: Path | None = None
    simpan_checkpoint: bool | None = None


_pekerja: _KonteksPekerja | None = None


def _awal_pekerja(
    panel: PanelData,
    lipatan: list[Fold],
    lookback: int,
    mk: dict[str, Any],
    tk: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    simpan_checkpoint: bool,
    thread: int,
) -> None:
    """Inisialisasi proses pekerja paralel."""
    global _pekerja
    torch.set_num_threads(thread)
    _pekerja = _KonteksPekerja()
    _pekerja.panel = panel
    _pekerja.lipatan = lipatan
    _pekerja.lookback = lookback
    _pekerja.mk = mk
    _pekerja.tk = tk
    _pekerja.device = device
    _pekerja.run_dir = run_dir
    _pekerja.simpan_checkpoint = simpan_checkpoint


def _tugas_fit(pasangan: tuple[int, int]) -> tuple[int, int]:
    """Jalankan satu (lipatan, seed) di proses pekerja."""
    id_lipatan, seed = pasangan
    fold = _pekerja.lipatan[id_lipatan]
    jalankan_fit(
        _pekerja.panel,
        fold,
        _pekerja.lookback,
        _pekerja.mk,
        _pekerja.tk,
        seed,
        _pekerja.device,
        _pekerja.run_dir,
        _pekerja.simpan_checkpoint,
    )
    return id_lipatan, seed


def run_walkforward(args: argparse.Namespace) -> int:
    """Jalankan walk-forward penuh (atau smoke) dan tulis prediksi."""
    model_cfg = load_config("model")
    split_cfg = load_config("split")
    tickers = list(load_config("universe")["tickers"])

    if args.mode == "smoke":
        seeds = seed_list(args.seeds)[:1]
        batas_lipatan = args.folds or 2
        batas_epoch = args.epochs or 3
    else:
        seeds = seed_list(args.seeds)
        batas_lipatan = args.folds
        batas_epoch = args.epochs

    hp = muat_tuning(args.use_tuning)
    lookback = int(hp.get("lookback_days", model_cfg["lookback_days"]))
    horizon = int(hp.get("horizon_days", model_cfg["horizon_days"]))
    mk = model_kwargs(model_cfg, hp)
    tk = train_kwargs(model_cfg, hp, batas_epoch)

    device = resolve_device(model_cfg["device"])
    threads = args.threads or torch.get_num_threads()
    if args.jobs > 1 and not args.threads:
        threads = max(1, min(4, (os.cpu_count() or 1) // args.jobs))
    torch.set_num_threads(threads)

    panel = muat_panel(tickers, horizon)
    lipatan = make_folds(
        len(panel.dates),
        split_cfg,
        purge_days=horizon,
        embargo_days=lookback - 1,
    )
    if batas_lipatan:
        lipatan = lipatan[:batas_lipatan]

    run_dir = args.out or nama_run(args.mode)
    ensure_dirs(
        run_dir / "predictions_parts",
        run_dir / "checkpoints",
        run_dir / "metrics",
    )
    log(
        run_dir,
        f"mulai {args.mode}: {len(lipatan)} lipatan, {len(seeds)} seed, "
        f"jobs {args.jobs}, thread {threads}, perangkat {device}",
    )

    pasangan = [(fold.id, seed) for fold in lipatan for seed in seeds]
    pasangan = [
        p for p in pasangan if not (args.resume and sudah_selesai(run_dir, p[0], p[1]))
    ]
    if not pasangan:
        log(run_dir, "semua pasangan sudah selesai (mode lanjut)")
    elif args.jobs <= 1:
        for id_lipatan, seed in pasangan:
            fold = lipatan[id_lipatan]
            metrik = jalankan_fit(
                panel,
                fold,
                lookback,
                mk,
                tk,
                seed,
                device,
                run_dir,
                not args.no_checkpoints,
            )
            log(
                run_dir,
                f"lipatan {fold.id} seed {seed}: val "
                f"{metrik['val_loss']:.6f}, test {metrik['test_loss']:.6f}, "
                f"waktu {metrik['wall_sec']} detik",
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_awal_pekerja,
            initargs=(
                panel,
                lipatan,
                lookback,
                mk,
                tk,
                device,
                run_dir,
                not args.no_checkpoints,
                max(1, threads),
            ),
        ) as pool:
            for id_lipatan, seed in pool.map(_tugas_fit, pasangan):
                log(run_dir, f"selesai lipatan {id_lipatan} seed {seed}")

    prediksi = gabungkan(run_dir)
    tulis_run_info(
        run_dir,
        args.mode,
        seeds,
        args.jobs,
        threads,
        lookback,
        horizon,
        len(lipatan),
        str(device),
        args.use_tuning,
    )
    log(
        run_dir,
        f"selesai: {len(prediksi)} baris prediksi -> " f"{run_dir / 'predictions.csv'}",
    )
    if args.mode == "smoke":
        ringkasan_smoke(run_dir, prediksi)
    return 0


def main() -> int:
    """Jalankan mode yang diminta."""
    args = parse_args()
    if args.mode == "tune":
        return run_tune(args)
    if args.mode == "calibrate":
        return run_calibrate(args)
    if args.mode == "dry-run":
        return run_dryrun(args)
    if args.mode == "pretrain":
        return run_pretrain(args)
    if args.mode == "walk-forward-pretrain":
        return run_walkforward_pretrain(args)
    return run_walkforward(args)


if __name__ == "__main__":
    raise SystemExit(main())
