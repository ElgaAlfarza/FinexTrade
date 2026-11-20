"""Perhitungan ukuran lot dari risiko per trade dan manajemen proteksi modal."""
from __future__ import annotations

import math


def lot_size(
    equity: float,
    risk_pct: float,
    sl_pips: float,
    pip_value_per_lot: float,
    min_lot: float = 0.01,
    lot_step: float = 0.01,
    max_lot: float = 100.0,
) -> float:
    """Lot = (Ekuitas x Risiko%) / (Jarak SL dalam pip x Nilai per pip per lot).

    Dibulatkan ke BAWAH ke kelipatan lot_step supaya risiko tidak melebihi batas.
    Mengembalikan 0.0 jika lot minimum broker sudah melebihi batas risiko.
    """
    if equity <= 0 or risk_pct <= 0 or sl_pips <= 0 or pip_value_per_lot <= 0:
        raise ValueError("equity, risk_pct, sl_pips, dan pip_value_per_lot harus positif")
    raw = equity * (risk_pct / 100.0) / (sl_pips * pip_value_per_lot)
    lots = math.floor(raw / lot_step + 1e-9) * lot_step
    if lots < min_lot:
        return 0.0
    return round(min(lots, max_lot), 8)


def can_trade_today(daily_r: float, max_daily_loss_r: float) -> bool:
    """Circuit Breaker: return True jika belum mencapai batas rugi harian."""
    return daily_r > -abs(max_daily_loss_r)


def estimate_pip_value(symbol: str, pip_size: float = 0.0001, account_currency: str = "USD") -> float:
    """Estimasi nilai per pip untuk 1 lot standar (fallback saat order_calc_profit MT5 tidak aktif)."""
    sym = symbol.upper()
    if "BTC" in sym:
        # 1 lot BTC = 1 BTC. Jika pip_size = 1.0, maka $1 bergerak = $1 keuntungan per lot
        return 1.0
    elif "ETH" in sym:
        # 1 lot ETH = 1 ETH. Pip 0.10 -> $0.10 per 0.10 poin (atau $1 per $1)
        return max(0.1, float(pip_size))
    elif "US100" in sym or "US500" in sym:
        # Finex MT5 Index CFD: kontrak 20 unit -> 1.0 poin = $20 / lot
        return 20.0
    elif "US30" in sym:
        return 5.0
    elif "XTI" in sym or "USO" in sym:
        # Crude Oil 1,000 barel: pip 0.01 -> $10 / pip
        return 10.0
    elif "XAU" in sym or "GOLD" in sym:
        # XAUUSD: 1 lot = 100 oz. Pip = 0.10 -> $10 / pip
        return 10.0
    elif "XAG" in sym or "SILVER" in sym:
        # XAGUSD: 1 lot = 5,000 oz. Pip = 0.01 -> $50 / pip
        return 50.0
    elif sym.startswith("USD") and sym.endswith("JPY"):
        # USDJPY: pip = 0.01 -> ~ $6.67 / pip pada kurs ~150
        return 6.67
    elif sym.endswith("USD"):
        # EURUSD, GBPUSD, AUDUSD: 1 lot = 100,000 * 0.0001 = $10 / pip
        return 10.0
    return 10.0
