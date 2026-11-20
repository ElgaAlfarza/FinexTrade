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


def generate_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Kembalikan DataFrame side (1 buy, -1 sell, 0 tidak ada), sl, tp per candle."""
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
