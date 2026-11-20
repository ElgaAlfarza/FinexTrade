"""Mesin sinyal.

Alur keputusan:
1. Data entry di-resample menjadi timeframe tren (HTF).
2. Di HTF dicari swing high/low yang sudah terkonfirmasi, lalu tren ditentukan
   dari struktur (HH+HL = naik, LH+LL = turun, selain itu ranging).
3. Dari leg impuls terakhir ditarik Fibonacci. Zona masuk = retracement 50% sampai 78.6%.
   (Opsional) Filter MA di HTF: buy hanya jika MA mendukung naik, sell hanya jika mendukung turun.
4. Di timeframe entry, sinyal muncul jika candle menyentuh zona dan menutup
   sebagai candle rejection searah tren.

Semua keputusan hanya memakai candle yang sudah tertutup (tanpa lookahead).
Sinyal dihitung pada penutupan candle i; eksekusi terjadi di open candle i+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import StrategyParams

_STATE_COLS = ["trend", "buy_L", "buy_H", "sell_H", "sell_L"]


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = df[list(agg)].resample(rule, label="left", closed="left").agg(agg)
    return out.dropna()


def _add_pivot(pivots: list, typ: int, idx: int, price: float) -> bool:
    """Tambah pivot dengan aturan selang-seling. True jika daftar berubah."""
    if pivots and pivots[-1][0] == typ:
        more_extreme = price > pivots[-1][2] if typ == 1 else price < pivots[-1][2]
        if more_extreme:
            pivots[-1] = (typ, idx, price)
            return True
        return False
    pivots.append((typ, idx, price))
    del pivots[:-8]  # cukup simpan 8 pivot terakhir
    return True


def _leg(pivots: list, typ: int):
    """Pivot bertipe typ yang terakhir beserta pivot sebelumnya (awal leg)."""
    for pos in range(len(pivots) - 1, 0, -1):
        if pivots[pos][0] == typ:
            return pivots[pos - 1][2], pivots[pos][2]
    return np.nan, np.nan


def _state_from_pivots(pivots: list):
    highs = [p[2] for p in pivots if p[0] == 1][-2:]
    lows = [p[2] for p in pivots if p[0] == -1][-2:]
    trend = 0
    if len(highs) == 2 and len(lows) == 2:
        if highs[1] > highs[0] and lows[1] > lows[0]:
            trend = 1
        elif highs[1] < highs[0] and lows[1] < lows[0]:
            trend = -1
    buy_L, buy_H = _leg(pivots, 1)       # leg naik: low lalu high
    sell_H, sell_L = _leg(pivots, -1)    # leg turun: high lalu low
    return (trend, buy_L, buy_H, sell_H, sell_L)


def htf_structure(htf: pd.DataFrame, k: int) -> pd.DataFrame:
    """Status struktur per candle HTF. Swing di indeks j baru terkonfirmasi di j+k."""
    h = htf["high"].to_numpy(float)
    l = htf["low"].to_numpy(float)
    n = len(htf)
    out = np.full((n, 5), np.nan)
    pivots: list = []
    state = (np.nan,) * 5
    for i in range(n):
        j = i - k
        if j >= k:
            wh = h[j - k : i + 1]
            wl = l[j - k : i + 1]
            is_high = h[j] == wh.max() and int((wh == h[j]).sum()) == 1
            is_low = l[j] == wl.min() and int((wl == l[j]).sum()) == 1
            if is_high != is_low:
                typ, price = (1, h[j]) if is_high else (-1, l[j])
                if _add_pivot(pivots, typ, j, price):
                    state = _state_from_pivots(pivots)
        out[i] = state
    return pd.DataFrame(out, index=htf.index, columns=_STATE_COLS)


def align_htf(df: pd.DataFrame, htf_info: pd.DataFrame) -> pd.DataFrame:
    """Pasangkan status HTF ke candle entry.

    shift(1): status baris HTF ke-s hanya memakai candle HTF sebelumnya yang
    sudah tertutup pada atau sebelum waktu mulai baris s.
    """
    return htf_info.shift(1).reindex(df.index, method="ffill")


def moving_average(close: pd.Series, period: int, ma_type: str = "EMA") -> pd.Series:
    """MA sederhana (SMA) atau eksponensial (EMA). Hanya memakai data sampai candle ke-i."""
    if period < 1:
        raise ValueError("periode MA harus >= 1")
    t = ma_type.upper()
    if t == "SMA":
        return close.rolling(period, min_periods=period).mean()
    if t == "EMA":
        return close.ewm(span=period, adjust=False, min_periods=period).mean()
    raise ValueError("ma_type harus 'SMA' atau 'EMA'")


def ma_htf_frame(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Hitung MA pada timeframe tren (HTF) lalu pasangkan ke candle entry.

    Memakai align_htf (shift(1)): candle entry hanya melihat nilai MA dari candle HTF yang
    SUDAH tertutup, jadi tidak ada lookahead. Kolom: close, fast, slow (NaN jika belum cukup data).
    """
    htf = resample_ohlc(df, p.htf_rule)
    info = pd.DataFrame(index=htf.index)
    info["htf_close"] = htf["close"]
    info["ma_fast"] = moving_average(htf["close"], p.ma_fast, p.ma_type) if p.ma_fast and p.ma_fast > 0 else np.nan
    info["ma_slow"] = moving_average(htf["close"], p.ma_slow, p.ma_type)
    return align_htf(df, info)


def ma_direction(ma: pd.DataFrame, mode: str) -> np.ndarray:
    """+1 jika MA mendukung buy, -1 jika mendukung sell, 0 jika netral/belum siap."""
    c = ma["htf_close"].to_numpy(float)
    f = ma["ma_fast"].to_numpy(float)
    s = ma["ma_slow"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        if mode == "price_slow":
            up, dn = c > s, c < s
        elif mode == "cross":
            up, dn = f > s, f < s
        elif mode == "price_both":
            up, dn = (c > s) & (c > f), (c < s) & (c < f)
        else:
            raise ValueError("ma_mode harus 'price_slow', 'cross', atau 'price_both'")
    return np.where(up, 1, np.where(dn, -1, 0))


def generate_ema_trend_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Sinyal EMA 8/21/50 Trend Following + ADX + RSI + Dynamic Pocket (100% identik dengan live)."""
    pip = p.pip_size
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    n = len(df)
    if n < 50:
        return pd.DataFrame({"side": np.zeros(n, dtype=int), "sl": np.full(n, np.nan), "tp": np.full(n, np.nan)}, index=df.index)

    # 1. EMA 8, 21, 50
    ema8_series = df["close"].ewm(span=8, adjust=False).mean()
    ema21_series = df["close"].ewm(span=21, adjust=False).mean()
    ema50_series = df["close"].ewm(span=50, adjust=False).mean()

    ema8 = ema8_series.to_numpy()
    ema21 = ema21_series.to_numpy()
    ema50 = ema50_series.to_numpy()

    ema8_slope = np.zeros(n)
    ema21_slope = np.zeros(n)
    ema8_slope[1:] = (ema8[1:] - ema8[:-1]) / pip
    ema21_slope[1:] = (ema21[1:] - ema21[:-1]) / pip

    # 2. True Range & ATR (14)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr_series = pd.Series(tr, index=df.index).rolling(14, min_periods=1).mean()
    atr = atr_series.to_numpy()
    atr = np.maximum(atr, 2.0 * pip)

    # 3. RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi_series.to_numpy()

    # 4. ADX (14)
    up_move = np.zeros(n)
    down_move = np.zeros(n)
    up_move[1:] = h[1:] - h[:-1]
    down_move[1:] = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_smooth = pd.Series(tr, index=df.index).ewm(alpha=1/14, adjust=False).mean().to_numpy()
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean().to_numpy() / (tr_smooth + 1e-9))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean().to_numpy() / (tr_smooth + 1e-9))
    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = pd.Series(dx, index=df.index).ewm(alpha=1/14, adjust=False).mean().to_numpy()

    # 5. Swing High/Low 20
    swing_h20 = pd.Series(h, index=df.index).shift(1).rolling(20, min_periods=1).max().to_numpy()
    swing_l20 = pd.Series(l, index=df.index).shift(1).rolling(20, min_periods=1).min().to_numpy()

    candle_body = c - o
    candle_range = np.maximum(1e-9, h - l)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l

    # Alignment
    is_bull_align = (ema8 > ema21) & (ema21 > ema50)
    is_bear_align = (ema8 < ema21) & (ema21 < ema50)

    # Pocket & Rejection
    bull_in_pocket = (l <= ema8 + 1.2 * pip) & (c >= ema21 - 2.0 * pip)
    bear_in_pocket = (h >= ema8 - 1.2 * pip) & (c <= ema21 + 2.0 * pip)

    valid_candle_size = candle_range >= (0.45 * atr)
    is_bull_rejection = valid_candle_size & (
        (lower_wick >= 0.35 * candle_range) |
        ((candle_body > 0) & ((c - o) >= 0.50 * candle_range) & (c > ema8))
    )
    is_bear_rejection = valid_candle_size & (
        (upper_wick >= 0.35 * candle_range) |
        ((candle_body < 0) & ((o - c) >= 0.50 * candle_range) & (c < ema8))
    )

    # Direction preference filter
    direction_pref = str(getattr(p, "scalping_direction", "BOTH") or "BOTH").upper()
    allow_buy = direction_pref in ("BOTH", "BUY_ONLY")
    allow_sell = direction_pref in ("BOTH", "SELL_ONLY")

    # Filters
    buy_ok = allow_buy & (
        is_bull_align &
        (adx >= 20.0) &
        (ema21_slope >= 0.20) &
        (rsi <= 68.0) &
        (swing_h20 - c >= 1.0 * atr) &
        ((c - ema8) <= 2.0 * atr) &
        bull_in_pocket &
        is_bull_rejection
    )

    sell_ok = allow_sell & (
        is_bear_align &
        (adx >= 20.0) &
        (ema21_slope <= -0.20) &
        (rsi >= 32.0) &
        (c - swing_l20 >= 1.0 * atr) &
        ((ema8 - c) <= 2.0 * atr) &
        bear_in_pocket &
        is_bear_rejection
    )

    side = np.where(buy_ok, 1, np.where(sell_ok, -1, 0))
    side[:50] = 0

    target_usd = float(getattr(p, "target_profit_usd", 5.0) or 5.0)
    loss_usd = float(getattr(p, "target_loss_usd", 5.0) or 5.0)
    lot = float(getattr(p, "scalping_lot", 0.1) or 0.1)
    # 0.1 lot on Gold ($1 per pip): $5 target = 5 pips
    target_pips = max(5.0, target_usd / (lot * 10.0) if lot > 0 else 10.0)
    loss_pips = max(5.0, loss_usd / (lot * 10.0) if lot > 0 else 10.0)

    sl = np.where(side == 1, c - loss_pips * pip, np.where(side == -1, c + loss_pips * pip, np.nan))
    tp = np.where(side == 1, c + target_pips * pip, np.where(side == -1, c - target_pips * pip, np.nan))

    return pd.DataFrame({"side": side, "sl": sl, "tp": tp}, index=df.index)


def _generate_fibo_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Implementasi Fibonacci Swing Structure (Tahap 2)."""
    htf = resample_ohlc(df, p.htf_rule)
    info = align_htf(df, htf_structure(htf, p.pivot_k))

    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    t = info["trend"].to_numpy()
    bL, bH = info["buy_L"].to_numpy(), info["buy_H"].to_numpy()
    sH, sL = info["sell_H"].to_numpy(), info["sell_L"].to_numpy()

    buf = p.sl_buffer_pips * p.pip_size
    rng = h - l

    with np.errstate(invalid="ignore"):
        lower_wick = np.minimum(o, c) - l
        upper_wick = h - np.maximum(o, c)
        bull = (c > o) & (rng > 0) & (lower_wick >= p.wick_ratio * rng)
        bear = (c < o) & (rng > 0) & (upper_wick >= p.wick_ratio * rng)

        size_b = bH - bL
        zone_top_b = bH - p.fib_top * size_b        # 50%
        zone_bot_b = bH - p.fib_bottom * size_b     # 78.6%
        buy = (
            (t == 1) & (size_b > 0) & bull
            & (l <= zone_top_b) & (c >= zone_bot_b) & (l > bL) & (c < bH)
        )

        size_s = sH - sL
        zone_bot_s = sL + p.fib_top * size_s        # 50%
        zone_top_s = sL + p.fib_bottom * size_s     # 78.6%
        sell = (
            (t == -1) & (size_s > 0) & bear
            & (h >= zone_bot_s) & (c <= zone_top_s) & (h < sH) & (c > sL)
        )

        # Filter Moving Average (opsional): sinyal hanya lolos jika MA HTF searah
        if getattr(p, "use_ma_filter", False):
            mdir = ma_direction(ma_htf_frame(df, p), p.ma_mode)
            buy = buy & (mdir == 1)
            sell = sell & (mdir == -1)

        side = np.where(buy, 1, np.where(sell, -1, 0))

        # Filter sesi jika diaktifkan (Trader best practice)
        if getattr(p, "use_session_filter", False):
            hours = df.index.hour
            in_session = (hours >= p.session_start_utc) & (hours < p.session_end_utc)
            side = np.where(in_session, side, 0)
            buy = buy & in_session
            sell = sell & in_session

        if p.sl_mode == "structure":
            sl = np.where(buy, bL - buf, np.where(sell, sH + buf, np.nan))
        elif p.sl_mode == "candle":
            sl = np.where(buy, l - buf, np.where(sell, h + buf, np.nan))
        else:
            raise ValueError("sl_mode harus 'structure' atau 'candle'")
        tp = np.where(
            buy, bH + p.tp_extension * size_b,
            np.where(sell, sL - p.tp_extension * size_s, np.nan),
        )

    return pd.DataFrame({"side": side, "sl": sl, "tp": tp}, index=df.index)


def generate_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Kembalikan DataFrame side (1 buy, -1 sell, 0 tidak ada), sl, tp per candle."""
    tech = getattr(p, "active_technique", "FIBO")
    sl_mode = getattr(p, "sl_mode", "")
    if tech == "EMA_TREND" and sl_mode not in ("structure", "candle"):
        return generate_ema_trend_signals(df, p)
    return _generate_fibo_signals(df, p)
