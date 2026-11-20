"""Skrip Evaluasi Backtest In-Sample vs Out-of-Sample Strategi Live (EMA_TREND).

Sesuai Aturan Keras & Kejujuran Kuantitatif:
1. Tidak ada curve-fitting (parameter tidak diotak-atik demi hasil bagus).
2. Memakai data historis nyata XAUUSD M5 broker Finex (25.000 candle).
3. Split 70% In-Sample (latih) vs 30% Out-of-Sample (uji).
4. Biaya transaksi nyata (spread 1.8 pip + slippage 0.5 pip).
5. Hasil jelek dilaporkan apa adanya dengan interval kepercayaan 95% Wilson.
"""
from __future__ import annotations

import json
import os
import sys
import pandas as pd

BOT_DIR = os.path.dirname(__file__)
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

from backtest import run_walk_forward_backtest, with_params
from config import build_params, StrategyParams


def run_evaluation(csv_path: str = "") -> dict:
    if not csv_path:
        csv_path = os.path.join(BOT_DIR, "data", "XAUUSD_M5.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File data historis tidak ditemukan di {csv_path}")

    raw = pd.read_csv(csv_path)
    raw["time"] = pd.to_datetime(raw["time"])
    df = raw.set_index("time")[["open", "high", "low", "close"]].sort_index()

    p = build_params("XAUUSD", "M5")
    p = with_params(
        p,
        active_technique="EMA_TREND",
        sl_mode="atr",
        min_rr=0.8,
        spread_pips=1.8,
        slippage_pips=0.5,
        target_profit_usd=5.0,
        target_loss_usd=5.0,
        risk_pct=1.0,
    )

    res = run_walk_forward_backtest(df, p, train_ratio=0.7)

    report_path = os.path.join(BOT_DIR, "data", "train_test_evaluation_report.json")
    train_st = res["train_stats"]
    test_st = res["test_stats"]

    save_payload = {
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "technique": "EMA_TREND",
        "total_candles": len(df),
        "train_range": res["train_range"],
        "test_range": res["test_range"],
        "in_sample": {
            "trades": train_st.get("trades", 0),
            "win_rate": train_st.get("win_rate", 0.0),
            "win_rate_ci95": train_st.get("win_rate_ci95", [0.0, 0.0]),
            "profit_factor": train_st.get("profit_factor", 0.0),
            "expectancy_r": train_st.get("expectancy_r", 0.0),
            "max_drawdown": train_st.get("max_drawdown", 0.0),
            "end_equity": train_st.get("end_equity", 10000.0),
        },
        "out_of_sample": {
            "trades": test_st.get("trades", 0),
            "win_rate": test_st.get("win_rate", 0.0),
            "win_rate_ci95": test_st.get("win_rate_ci95", [0.0, 0.0]),
            "profit_factor": test_st.get("profit_factor", 0.0),
            "expectancy_r": test_st.get("expectancy_r", 0.0),
            "max_drawdown": test_st.get("max_drawdown", 0.0),
            "end_equity": test_st.get("end_equity", 10000.0),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    return res


if __name__ == "__main__":
    res = run_evaluation()
    print("=" * 70)
    print("HASIL EVALUASI STRATEGI LIVE (EMA_TREND) PADA DATA HISTORIS XAUUSD M5")
    print("=" * 70)
    print("\n[1] HASIL IN-SAMPLE (70% Data Awal: {} s/d {})".format(*res["train_range"]))
    print(res["train_report"])
    print("\n" + "-" * 70)
    print("[2] HASIL OUT-OF-SAMPLE (30% Data Uji: {} s/d {})".format(*res["test_range"]))
    print(res["test_report"])
    print("=" * 70)
