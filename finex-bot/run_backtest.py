"""Jalankan backtest dari file CSV hasil ekspor MT5.

Contoh:
    python run_backtest.py --csv data/EURUSD_M5.csv --tf M5
    python run_backtest.py --csv data/EURUSD_M1.csv --tf M1 --sl-mode candle --be
"""
from __future__ import annotations

import argparse

import pandas as pd

from backtest import compute_stats, format_report, run_backtest, with_params
from config import PRESETS, StrategyParams

REQUIRED = ["time", "open", "high", "low", "close"]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Kolom tidak ditemukan di CSV: {missing}")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df[~df.index.duplicated(keep="first")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", choices=list(PRESETS), required=True, help="timeframe entry data CSV")
    ap.add_argument("--sl-mode", choices=["structure", "candle"], default="structure")
    ap.add_argument("--be", action="store_true", help="aktifkan auto Break-Even saat floating profit mencapai 1R")
    ap.add_argument("--htf-rule", help="timpa timeframe tren, contoh 30min atau 1h")
    ap.add_argument("--pip-size", type=float, default=0.0001)
    ap.add_argument("--spread-pips", type=float, default=0.8)
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--ma", action="store_true", help="aktifkan filter Moving Average di timeframe tren")
    ap.add_argument("--ma-type", choices=["SMA", "EMA"], default="EMA")
    ap.add_argument("--ma-fast", type=int, default=21, help="periode MA cepat (0 = tidak dipakai)")
    ap.add_argument("--ma-slow", type=int, default=50, help="periode MA lambat")
    ap.add_argument("--ma-mode", choices=["price_slow", "cross", "price_both"], default="price_slow")
    ap.add_argument("--out", help="simpan daftar trade ke CSV")
    a = ap.parse_args()

    kw = dict(PRESETS[a.tf])
    if a.htf_rule:
        kw["htf_rule"] = a.htf_rule
    p = with_params(
        StrategyParams(), sl_mode=a.sl_mode, pip_size=a.pip_size,
        spread_pips=a.spread_pips, risk_pct=a.risk_pct, use_break_even=a.be,
        use_ma_filter=a.ma, ma_type=a.ma_type, ma_fast=a.ma_fast, ma_slow=a.ma_slow, ma_mode=a.ma_mode, **kw,
    )

    df = load_csv(a.csv)
    print(f"Data: {len(df)} candle, {df.index[0]} sampai {df.index[-1]}")
    print(f"Timeframe entry {a.tf}, tren {p.htf_rule}, SL {p.sl_mode}, Break-Even: {'ON' if a.be else 'OFF'}, MA: {(a.ma_type + ' ' + str(a.ma_fast) + '/' + str(a.ma_slow) + ' ' + a.ma_mode) if a.ma else 'OFF'}, spread {p.spread_pips} pip\n")
    trades = run_backtest(df, p)
    print(format_report(compute_stats(trades, p), p))
    if a.out and not trades.empty:
        trades.to_csv(a.out, index=False)
        print(f"\nDaftar trade disimpan ke {a.out}")


if __name__ == "__main__":
    main()
