"""Generator data historis pasar realistis (EURUSD M1, M5, H1 & XAUUSD M5)
untuk keperluan pengujian backtest langsung dan visualisasi dashboard.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def get_default_base_price(symbol: str) -> float:
    sym = symbol.upper()
    if "BTC" in sym: return 76500.0
    if "ETH" in sym: return 2950.0
    if "XAU" in sym or "GOLD" in sym: return 2735.0
    if "XAG" in sym or "SILVER" in sym: return 32.40
    if "JPY" in sym: return 153.50
    if "GBP" in sym: return 1.2950
    if "AUD" in sym: return 0.6550
    return 1.0850


def generate_market_data(
    symbol: str = "EURUSD",
    timeframe: str = "M5",
    bars: int = 25000,
    base_price: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sym = symbol.upper()

    if base_price is None or base_price == 1.0850:
        base_price = get_default_base_price(sym)

    # Tentukan frekuensi waktu
    freq_map = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1d"}
    freq = freq_map.get(timeframe, "5min")

    end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    # Buat indeks tanggal ke belakang
    date_index = pd.date_range(end=end_time, periods=bars, freq=freq)

    # Model multi-regime: bull trending, bear trending, ranging
    regime_lengths = rng.integers(200, 800, size=bars // 200 + 1)
    regimes = []

    if "BTC" in sym:
        regime_choices = [5.0, -5.0, 0.0]
        base_vol = 25.0
        decimals = 2
    elif "ETH" in sym:
        regime_choices = [0.3, -0.3, 0.0]
        base_vol = 1.5
        decimals = 2
    elif "XAU" in sym:
        regime_choices = [0.08, -0.08, 0.0]
        base_vol = 0.45
        decimals = 2
    elif "XAG" in sym:
        regime_choices = [0.01, -0.01, 0.0]
        base_vol = 0.05
        decimals = 2
    elif "JPY" in sym:
        regime_choices = [0.02, -0.02, 0.0]
        base_vol = 0.08
        decimals = 3
    else:
        regime_choices = [0.00004, -0.00004, 0.0]
        base_vol = 0.00015
        decimals = 5

    for length in regime_lengths:
        r = rng.choice(regime_choices, p=[0.35, 0.35, 0.30])
        regimes.extend([r] * length)
    regimes = np.array(regimes[:bars])

    # Volatilitas berdasarkan jam sesi
    hours = date_index.hour
    vol_multiplier = np.where((hours >= 7) & (hours <= 20), 1.4, 0.8)

    vol = base_vol * vol_multiplier * (1.0 + 0.2 * rng.standard_normal(bars))
    vol = np.maximum(vol, base_vol * 0.3)

    delta_steps = regimes + vol * rng.standard_normal(bars)
    close_prices = base_price + np.cumsum(delta_steps)

    # Bangun Open, High, Low, Close
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    wick_up = np.abs(rng.standard_normal(bars)) * vol * 0.8
    wick_down = np.abs(rng.standard_normal(bars)) * vol * 0.8

    high_prices = np.maximum(open_prices, close_prices) + wick_up
    low_prices = np.minimum(open_prices, close_prices) - wick_down

    # Format dataframe sesuai standar ekspor MT5
    df = pd.DataFrame({
        "time": date_index.strftime("%Y-%m-%d %H:%M:%S"),
        "open": np.round(open_prices, decimals),
        "high": np.round(high_prices, decimals),
        "low": np.round(low_prices, decimals),
        "close": np.round(close_prices, decimals),
        "tick_volume": rng.integers(50, 1500, size=bars),
        "spread": np.repeat(int(10 if decimals == 5 else 25), bars),
        "real_volume": np.zeros(bars, dtype=int),
    })

    return df


def main():
    os.makedirs("data", exist_ok=True)
    datasets = [
        ("EURUSD", "M5", 30000, 1.0850, "data/EURUSD_M5.csv"),
        ("EURUSD", "M1", 40000, 1.0850, "data/EURUSD_M1.csv"),
        ("EURUSD", "H1", 8000, 1.0850, "data/EURUSD_H1.csv"),
        ("XAUUSD", "M5", 25000, 2750.00, "data/XAUUSD_M5.csv"),
    ]

    for sym, tf, bars, base, out in datasets:
        print(f"Menghasilkan {bars} candle {sym} {tf}...")
        df = generate_market_data(sym, tf, bars, base)
        df.to_csv(out, index=False)
        print(f"-> Disimpan ke {out} ({len(df)} baris)")


if __name__ == "__main__":
    main()
