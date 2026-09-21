#!/usr/bin/env python3
"""Tahap 5: evaluasi risiko, signifikansi, dan ketahanan rezim.

Masukan: keluaran tahap 4 (`portfolio_returns.csv`,
`baseline_returns.csv`, `weights.csv`).
Keluaran (`--out`, bawaan `reports/`):
- `metrics_table.csv`    : metrik per konfigurasi MV (penuh dan OOS)
- `regime_table.csv`     : metrik per rezim (COVID 2020, 2021-2025)
- `seed_spread.csv`      : sebaran Sharpe lintas seed per (k, estimator)
- `significance.csv`     : uji HAC dan Romano-Wolf lawan pembanding
- `dsr_pbo.json`         : DSR strategi terbaik dan PBO grid utama
- `figures/`             : kurva ekuitas, drawdown, perbandingan Sharpe

Jejak keputusan: docs/keputusan_desain.md bagian E dan H.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.evaluation.dsr_pbo import deflated_sharpe, pbo_cscv
from lq45.evaluation.metrics import (
    bebas_risiko_harian,
    ringkas_deret,
)
from lq45.evaluation.regimes import bagi_rezim
from lq45.evaluation.significance import (
    romano_wolf_stepdown,
    uji_beda_sharpe,
)
from lq45.utils.config import REPORT_DIR, load_config


def parse_args() -> argparse.Namespace:
    """Baca argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Evaluasi portofolio tahap 5.")
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def git_sha() -> str:
    """Hash pendek commit terakhir; `unknown` bila gagal."""
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


def kunci_konfig(frame: pd.DataFrame) -> pd.Series:
    """Kunci teks konfigurasi dari kolom grid."""
    return (
        frame["k"].astype(str)
        + "|"
        + frame["estimator"]
        + "|L"
        + frame["lookback"].astype(str)
        + "|m"
        + frame["max_weight"].astype(str)
        + "|"
        + frame["seed_mode"]
    )


def turnover_rerata(bobot: pd.DataFrame, kunci: str) -> float:
    """Rata-rata perputaran per rebalancing untuk satu konfigurasi.

    Perputaran = jumlah |selisih bobot| / 2 antar tanggal rebalancing
    berurutan (satu arah, pecahan portofolio).
    """
    potong = bobot[bobot["config"] == kunci].copy()
    if potong.empty:
        return 0.0
    pivot = potong.pivot_table(
        index="date", columns="ticker", values="weight", fill_value=0.0
    ).sort_index()
    if len(pivot) < 2:
        return 0.0
    selisih = pivot.diff().iloc[1:].abs().sum(axis=1) / 2.0
    return float(selisih.mean())


def main() -> int:
    """Hitung metrik, uji, dan figur lalu tulis keluaran."""
    args = parse_args()
    exp_cfg = load_config("experiment")
    tahun = int(exp_cfg["evaluation"]["annualization"])
    rezim_cfg = {k: v for k, v in exp_cfg["regimes"].items() if v.get("oos", False)}
    keluar = args.out or REPORT_DIR
    figur_dir = keluar / "figures"
    figur_dir.mkdir(parents=True, exist_ok=True)
    port = pd.read_csv(args.portfolio, parse_dates=["date"])
    dasar = pd.read_csv(args.baselines, parse_dates=["date"])
    bobot = pd.read_csv(args.weights, parse_dates=["date"])
    port["config"] = kunci_konfig(port)
    tanggal = pd.DatetimeIndex(sorted(port["date"].unique()))
    rf = bebas_risiko_harian(
        tanggal, str(ROOT / "data" / "raw" / "macro" / "bi_7drrr.csv"), tahun
    )
    rf.index = tanggal
    baris: list[dict] = []
    deret: dict[str, pd.Series] = {}
    bobot["config"] = kunci_konfig(bobot)
    for kunci, grup in port.groupby("config"):
        grup = grup.sort_values("date")
        imbal = grup.set_index("date")["return"]
        deret[kunci] = imbal
        ringkas = ringkas_deret(imbal, rf.reindex(imbal.index).fillna(0.0), tahun)
        putar = turnover_rerata(bobot, kunci)
        ringkas["turnover"] = putar
        ringkas["turnover_adjusted_sharpe"] = ringkas["sharpe"] * (1.0 - putar)
        ringkas["config"] = kunci
        ringkas.update(
            {
                "k": grup["k"].iloc[0],
                "estimator": grup["estimator"].iloc[0],
                "lookback": grup["lookback"].iloc[0],
                "max_weight": grup["max_weight"].iloc[0],
                "seed_mode": grup["seed_mode"].iloc[0],
            }
        )
        baris.append(ringkas)
    metrik = pd.DataFrame(baris).sort_values("sharpe", ascending=False)
    metrik.to_csv(keluar / "metrics_table.csv", index=False)
    # Rezim OOS per konfigurasi.
    baris_rezim: list[dict] = []
    for kunci, imbal in deret.items():
        frame = pd.DataFrame({"date": imbal.index, "return": imbal.to_numpy()})
        for nama, potong in bagi_rezim(frame, "date", rezim_cfg).items():
            if potong.empty:
                continue
            idx_r = pd.DatetimeIndex(pd.to_datetime(potong["date"]))
            ringkas = ringkas_deret(
                potong.set_index("date")["return"],
                rf.reindex(idx_r).fillna(0.0),
                tahun,
            )
            ringkas["config"] = kunci
            ringkas["regime"] = nama
            baris_rezim.append(ringkas)
    pd.DataFrame(baris_rezim).to_csv(keluar / "regime_table.csv", index=False)
    # Sebaran lintas seed per (k, estimator) pada grid utama.
    utama = metrik[(metrik["lookback"] == 120) & (metrik["max_weight"] == 0.35)]
    sebar = (
        utama.groupby(["k", "estimator"])["sharpe"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    sebar.to_csv(keluar / "seed_spread.csv", index=False)
    # Signifikansi: konfigurasi teratas lawan 1/N dan IHSG.
    dasar_idx = dasar.set_index("date")
    acuan_1n = dasar_idx["ret_1n_full"]
    acuan_ihsg = dasar_idx["ret_ihsg"]
    baris_uji: list[dict] = []
    for kunci in metrik.head(args.top_n)["config"]:
        imbal = deret[kunci]
        for nama_acuan, acuan in (("1N", acuan_1n), ("IHSG", acuan_ihsg)):
            uji = uji_beda_sharpe(imbal, acuan, tahun)
            uji["config"] = kunci
            uji["benchmark"] = nama_acuan
            baris_uji.append(uji)
    rw_masuk = utama[utama["seed_mode"] == "ensemble"].head(12)
    matriks_rw = pd.DataFrame({c: deret[c] for c in rw_masuk["config"]})
    tabel_rw = romano_wolf_stepdown(matriks_rw, acuan_1n, seed=0)
    tabel_rw["benchmark"] = "1N"
    pd.concat(
        [pd.DataFrame(baris_uji), tabel_rw], ignore_index=True, sort=False
    ).to_csv(keluar / "significance.csv", index=False)
    # DSR strategi terbaik dan PBO grid utama ensemble.
    terbaik = metrik.iloc[0]["config"]
    n_uji = float(len(metrik))
    dsr = deflated_sharpe(deret[terbaik], n_uji=int(n_uji))
    matriks_pbo = pd.DataFrame({c: deret[c] for c in rw_masuk["config"]})
    pbo = pbo_cscv(matriks_pbo, bagian=8, seed=0)
    (keluar / "dsr_pbo.json").write_text(
        json.dumps(
            {
                "best_config": terbaik,
                "n_trials": n_uji,
                "dsr": dsr,
                "pbo": pbo,
                "git_sha": git_sha(),
                "created_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Figur: ekuitas 5 teratas + pembanding, dan Sharpe grid utama.
    ekuitas = pd.DataFrame({c: (1.0 + deret[c]).cumprod() for c in deret})
    lima = metrik.head(5)["config"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in lima:
        ax.plot(ekuitas.index, ekuitas[c], label=c, linewidth=1.0)
    ax.plot(
        dasar_idx.index,
        (1.0 + dasar_idx["ret_1n_full"]).cumprod(),
        label="1/N",
        linestyle="--",
    )
    ax.plot(
        dasar_idx.index,
        (1.0 + dasar_idx["ret_ihsg"]).cumprod(),
        label="IHSG",
        linestyle=":",
    )
    ax.set_title("Kurva ekuitas: 5 konfigurasi teratas dan pembanding")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Pertumbuhan modal (x)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figur_dir / "equity_top5.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ens120 = utama[utama["seed_mode"] == "ensemble"].sort_values(["k", "estimator"])
    ax.bar(range(len(ens120)), ens120["sharpe"])
    ax.set_xticks(range(len(ens120)))
    ax.set_xticklabels(
        ens120["k"].astype(str) + "/" + ens120["estimator"], rotation=45, fontsize=7
    )
    ax.set_title("Sharpe grid utama ensemble (L120, maxw 0.35)")
    ax.set_ylabel("Sharpe tahunan")
    fig.tight_layout()
    fig.savefig(figur_dir / "sharpe_grid_utama.png", dpi=150)
    plt.close(fig)
    print(f"selesai: {len(metrik)} konfigurasi -> {keluar}", flush=True)
    print(f"terbaik: {terbaik}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
