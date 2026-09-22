#!/usr/bin/env python3
"""Stage 4: Top-K ranking and Mean-Variance optimization.

Input: `predictions.csv` from stage 3 (the `role` column marks the test set).
Outputs per run (`experiments/<run_id>/`):
    - `weights.csv`            : rebalancing weights per date and configuration
    - `portfolio_returns.csv`  : equity and daily net return per configuration
    - `baseline_returns.csv`   : 1/N pool and IHSG buy-and-hold
    - `run_info.json`          : execution record for auditing

Grid-search configuration (see `configs/portfolio.yaml`):
    - k: values {5, 7, 10, 3, 15, 20}, L: {120, 60, 252},
      max_weight: {0.35, 0.25, 0.50, 1.0}, and the ensemble as well as
      per-seed ranking modes. The `--grid main` option gives a fast
      run (k {5, 7, 10}, L 120, max_weight 0.35).

Methodology:
    Top-k selection uses the CNN-BiLSTM predictions. Mean-Variance
    optimization minimizes variance subject to a target return equal
    to the mean pred_ens (Chaweewanchon & Chaysiri 2022, Section 3.1).
    GMV is not used.

Decision log: docs/keputusan_desain.md sections E and H.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.portfolio.backtest import run_backtest
from lq45.portfolio.ranking import ensemble_predictions
from lq45.utils.config import (
    EXPERIMENT_DIR,
    INTERIM_DIR,
    ensure_dirs,
    load_config,
)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for the optimization configuration.

    Returns:
        Namespace with the parameters: predictions path, output directory,
        grid scope, initial capital, and seed ranking mode.
    """
    parser = argparse.ArgumentParser(
        description="Top-k ranking and portfolio optimization."
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="predictions.csv from stage 3"
    )
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--grid",
        choices=["main", "full"],
        default="full",
        help="grid scope (default: full)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000_000.0,
        help="initial capital in rupiah (default: 100 million)",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["ensemble", "all"],
        default="all",
        help="rank the ensemble only, or also per seed",
    )
    return parser.parse_args()


def git_sha() -> str:
    """Return the short (7-character) commit hash of the repository root.

    Returns:
        Short SHA hash string, or `"unknown"` if Git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_prices() -> pd.DataFrame:
    """Load the adjusted closing prices (Adj Close) of every LQ45 stock.

    Each CSV file in the interim directory represents one ticker.
    The index is dates and the columns are tickers with the `.JK` suffix.

    Returns:
        DataFrame with daily prices sorted by date.
    """
    frames: dict[str, pd.Series] = {}
    for csv_path in sorted((INTERIM_DIR / "prices_clean").glob("*.csv")):
        ticker = csv_path.stem + ".JK"
        frame = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
        frames[ticker] = frame["Adj Close"]
    prices = pd.DataFrame(frames).sort_index()
    prices.index = pd.to_datetime(prices.index)
    return prices


def prediction_frame(path: Path) -> pd.DataFrame:
    """Load the stage-3 prediction results and filter the test-set rows.

    Convert the `date` column to ISO 8601 strings and keep only rows with
    `role == "test"` for out-of-sample evaluation.

    Args:
        path: Path to the `predictions.csv` file.

    Returns:
        DataFrame with the test data and its index reset.
    """
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame[frame["role"] == "test"].reset_index(drop=True)


def ranking_frame(frame: pd.DataFrame, seed_mode: str) -> dict[str, pd.DataFrame]:
    """Build the prediction frames per ensemble mode and per seed.

    Each mode maps to a DataFrame with the columns `date`, `ticker`, and
    `pred_ens`. The `ensemble` mode averages all seeds; the individual-seed
    mode produces one DataFrame per seed for reporting the prediction spread.

    Args:
        frame: DataFrame produced by `prediction_frame`.
        seed_mode: `"ensemble"` averages all seeds; any other value keeps
            one frame per seed.

    Returns:
        Dictionary mapping the mode name to a ranking DataFrame.
    """
    ranking_frames: dict[str, pd.DataFrame] = {}
    ranking_frames["ensemble"] = ensemble_predictions(frame)
    if seed_mode == "all":
        for seed in sorted(frame["seed"].unique().tolist()):
            subset = frame[frame["seed"] == seed][["date", "ticker", "pred_raw"]]
            ranking_frames[f"seed{seed}"] = subset.rename(
                columns={"pred_raw": "pred_ens"}
            )
    return ranking_frames


def baselines(
    prices: pd.DataFrame, oos_dates: list[str], capital: float
) -> pd.DataFrame:
    """Compute the daily returns of the full-pool 1/N and IHSG benchmarks.

    1/N splits the capital equally across every available ticker on each
    date. IHSG is normalized to the initial capital on the first OOS date.

    Args:
        prices: Daily closing-price DataFrame.
        oos_dates: List of out-of-sample dates.
        capital: Initial capital in rupiah.

    Returns:
        DataFrame with the equity and daily returns of both benchmarks.
    """
    dates = pd.to_datetime(sorted(set(oos_dates)))
    oos_prices = prices[prices.index.isin(dates)].sort_index()
    returns = oos_prices.pct_change().fillna(0.0)
    ret_1n = returns.mean(axis=1)
    equity_1n = (1.0 + ret_1n).cumprod() * capital
    ihsg = pd.read_csv(
        ROOT / "data" / "raw" / "benchmark_ihsg.csv", parse_dates=["Date"]
    ).set_index("Date")
    ihsg.index = pd.to_datetime(ihsg.index)
    ihsg_oos = ihsg[ihsg.index.isin(dates)].sort_index()["Adj Close"]
    equity_ihsg = ihsg_oos / ihsg_oos.iloc[0] * capital
    merged = pd.DataFrame({"eq_1n_full": equity_1n, "eq_ihsg": equity_ihsg})
    merged.index.name = "date"
    merged = merged.reset_index()
    merged["ret_1n_full"] = merged["eq_1n_full"].pct_change().fillna(0.0)
    merged["ret_ihsg"] = merged["eq_ihsg"].pct_change().fillna(0.0)
    return merged


def main() -> int:
    """Run the portfolio-optimization grid search and write its results.

    Workflow:
        1. Load the configuration, prices, and predictions.
        2. Build the grid combinations (k, estimator, lookback, max_weight).
        3. Run the backtest per combination, appending results to CSV as
           it goes for resilience against failures.
        4. Compute the 1/N and IHSG baselines.
        5. Write the execution metrics to `run_info.json`.

    Returns:
        Exit code 0 on success.
    """
    args = parse_args()
    port_cfg = load_config("portfolio")
    data_cfg = load_config("data")
    opt_cfg = port_cfg["optimization"]
    start = time.time()
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out or (EXPERIMENT_DIR / f"optimize_{stamp}")
    ensure_dirs(run_dir)
    if args.grid == "main":
        k_list = list(port_cfg["selection"]["k_values"])
        l_list = [int(port_cfg["covariance"]["lookback_days"])]
        maxw_list = [float(port_cfg["constraints"]["max_weight"])]
    else:
        k_list = list(port_cfg["selection"]["k_values"]) + list(
            port_cfg["selection"]["k_sensitivity"]
        )
        l_list = [int(port_cfg["covariance"]["lookback_days"])] + list(
            port_cfg["covariance"]["lookback_sensitivity"]
        )
        maxw_list = [float(port_cfg["constraints"]["max_weight"])] + [
            float(x) for x in port_cfg["constraints"]["max_weight_sensitivity"]
        ]
    estimators = list(port_cfg["covariance"]["estimators"])
    optimizer_types = opt_cfg.get("types", ["target_return"])
    costs = data_cfg["costs"]
    frame = prediction_frame(args.predictions)
    prices = load_prices()
    ranking = ranking_frame(frame, args.seed_mode)
    oos_dates = sorted(frame["date"].unique().tolist())
    total = (
        len(k_list)
        * len(estimators)
        * len(l_list)
        * len(maxw_list)
        * len(ranking)
        * len(optimizer_types)
    )
    progress = 0
    for optimizer_type in optimizer_types:
        for mode_name, ranking_df in ranking.items():
            for k in k_list:
                for est in estimators:
                    for length in l_list:
                        for maxw in maxw_list:
                            progress += 1
                            weights, value = run_backtest(
                                ranking_df,
                                prices,
                                int(k),
                                est,
                                int(length),
                                float(maxw),
                                capital=args.capital,
                                buy_fee=float(costs["buy_fee"]),
                                sell_fee=float(costs["sell_fee"]),
                                lot=int(costs["lot_size"]),
                                ridge_epsilon=float(
                                    port_cfg["covariance"]["ridge_epsilon"]
                                ),
                                optimizer_type=optimizer_type,
                            )
                            weights["seed_mode"] = mode_name
                            value["k"] = int(k)
                            value["estimator"] = est
                            value["lookback"] = int(length)
                            value["max_weight"] = float(maxw)
                            value["seed_mode"] = mode_name
                            # Append the results of each iteration to
                            # prevent data loss if the process fails
                            # partway through.
                            weights_path = run_dir / "weights.csv"
                            returns_path = run_dir / "portfolio_returns.csv"
                            weights.to_csv(
                                weights_path,
                                mode="a",
                                header=not weights_path.exists(),
                                index=False,
                            )
                            value.to_csv(
                                returns_path,
                                mode="a",
                                header=not returns_path.exists(),
                                index=False,
                            )
                            if progress % 100 == 0 or progress == total:
                                print(
                                    f"[{progress}/{total}] {optimizer_type} {mode_name} "
                                    f"k={k} {est} L={length} maxw={maxw}",
                                    flush=True,
                                )
    baselines(prices, oos_dates, args.capital).to_csv(
        run_dir / "baseline_returns.csv", index=False
    )
    info = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "predictions": str(args.predictions),
        "grid": args.grid,
        "k_values": k_list,
        "estimators": estimators,
        "optimizer_types": optimizer_types,
        "lookbacks": l_list,
        "max_weights": maxw_list,
        "seed_modes": sorted(ranking),
        "capital": args.capital,
        "n_configs": total,
        "wall_sec": round(time.time() - start, 1),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"done: {total} configurations -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
