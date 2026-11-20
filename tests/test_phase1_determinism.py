"""Unit test untuk Fase 1: Determinisme, Fail-Closed, Timezone UTC, dan Anti-Repaint."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
FINEX_DIR = BASE_DIR / "finex-bot"
if str(FINEX_DIR) not in sys.path:
    sys.path.insert(0, str(FINEX_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tests.mock_mt5 import MockMT5
from live_engine import LiveBotEngine
from execution import MT5Executor, OrderManager, PaperExecutor
from mt5_client import MT5Session
from config import StrategyParams


def make_test_df(n_bars: int = 30, base_price: float = 2650.0) -> pd.DataFrame:
    """Buat dataframe lilin M1 sintetis terkalibrasi waktu UTC."""
    idx = pd.date_range("2026-11-20 00:00:00", periods=n_bars, freq="1min", tz="UTC")
    data = []
    p = base_price
    for i in range(n_bars):
        o = p
        h = o + 0.3
        l = o - 0.3
        c = o + 0.1
        data.append({"open": o, "high": h, "low": l, "close": c, "tick_volume": 50})
        p = c
    return pd.DataFrame(data, index=idx)


def test_fail_closed_under_20_bars():
    """Memastikan sinyal DITOLAK (action = WAIT) jika data lilin kurang dari 20 bar."""
    engine = LiveBotEngine(symbol="XAUUSD", timeframe="M1", use_paper=True)

    # 1. 0 bar
    sig_0 = engine._compute_active_signal(pd.DataFrame(), bid=2650.0, ask=2650.3)
    assert sig_0["action"] == "WAIT"
    assert "belum mencukupi" in sig_0["reason"]

    # 2. 15 bar (sebelumnya bisa memicu bug auto-BUY)
    df_15 = make_test_df(15)
    sig_15 = engine._compute_active_signal(df_15, bid=2650.0, ask=2650.3)
    assert sig_15["action"] == "WAIT"
    assert "belum mencukupi" in sig_15["reason"]

    # 3. 19 bar
    df_19 = make_test_df(19)
    sig_19 = engine._compute_active_signal(df_19, bid=2650.0, ask=2650.3)
    assert sig_19["action"] == "WAIT"
    assert "belum mencukupi" in sig_19["reason"]


def test_no_default_buy_on_flat_market():
    """Memastikan kondisi sideways / flat TIDAK menghasilkan sinyal BUY default hanya karena ask >= bid."""
    engine = LiveBotEngine(symbol="XAUUSD", timeframe="M1", use_paper=True)

    # 30 bar flat tanpa setup EMA ribbon
    idx = pd.date_range("2026-11-20 00:00:00", periods=30, freq="1min", tz="UTC")
    flat_df = pd.DataFrame({
        "open": [2650.0] * 30,
        "high": [2650.2] * 30,
        "low": [2649.8] * 30,
        "close": [2650.0] * 30,
        "tick_volume": [20] * 30,
    }, index=idx)

    # cur_ask >= cur_bid bernilai True
    sig = engine._compute_active_signal(flat_df, bid=2650.0, ask=2650.3)
    assert sig["action"] == "WAIT"
    assert sig["action"] != "BUY"


def test_anti_repainting_signal_stability():
    """Memastikan sinyal yang dihitung pada candle yang sudah tutup (closed_only=True) bersifat deterministik dan tidak berubah saat bar baru membentuk."""
    engine = LiveBotEngine(symbol="XAUUSD", timeframe="M1", use_paper=True)

    df_base = make_test_df(25, base_price=2650.0)
    sig_base = engine._compute_active_signal(df_base, bid=2650.0, ask=2650.3, closed_only=True)

    # Tambahkan bar ke-26 yang bergerak liar (forming bar)
    idx_new = pd.date_range("2026-11-20 00:25:00", periods=1, freq="1min", tz="UTC")
    df_forming = pd.concat([df_base, pd.DataFrame([{
        "open": 2650.0, "high": 2665.0, "low": 2640.0, "close": 2662.0, "tick_volume": 500
    }], index=idx_new)])

    # Sinyal closed_only pada bar forming harus tetap mengevaluasi bar closed yang sama
    sig_stable = engine._compute_active_signal(df_forming, bid=2662.0, ask=2662.3, closed_only=True)

    # Sinyal action harus konsisten
    assert sig_base["action"] == sig_stable["action"]


def test_mt5_executor_sl_required():
    """Memastikan MT5Executor menolak order tanpa Stop Loss (sl <= 0)."""
    mock_mt5 = MockMT5()
    session = MT5Session(login=123456, password="", server="Broker-Demo", require_demo=True)
    session.mt5 = mock_mt5
    executor = MT5Executor(session)

    # 1. SL = 0.0
    pos_no_sl = executor.open_trade("XAUUSD", "BUY", 0.01, sl=0.0, tp=2660.0, risk_pips=20.0)
    assert pos_no_sl is None
    assert "Stop Loss wajib" in executor.last_order_error

    # 2. SL negatif
    pos_neg_sl = executor.open_trade("XAUUSD", "BUY", 0.01, sl=-5.0, tp=2660.0, risk_pips=20.0)
    assert pos_neg_sl is None
    assert "Stop Loss wajib" in executor.last_order_error


def test_mt5_executor_lot_zero_rejected():
    """Memastikan MT5Executor menolak lot <= 0 dan tidak memaksakannya ke vmin."""
    mock_mt5 = MockMT5()
    session = MT5Session(login=123456, password="", server="Broker-Demo", require_demo=True)
    session.mt5 = mock_mt5
    executor = MT5Executor(session)

    pos_zero_lot = executor.open_trade("XAUUSD", "BUY", 0.0, sl=2640.0, tp=2660.0, risk_pips=20.0)
    assert pos_zero_lot is None
    assert "lot 0 atau negatif" in executor.last_order_error


def test_order_manager_fail_closed_zero_equity():
    """Memastikan OrderManager menolak order jika ekuitas akun <= 0 (tidak jatuh ke fallback $10.000)."""
    mock_mt5 = MockMT5()
    mock_mt5._account.equity = 0.0
    mock_mt5._account.balance = 0.0
    session = MT5Session(login=123456, password="", server="Broker-Demo", require_demo=True)
    session.mt5 = mock_mt5
    executor = MT5Executor(session)

    om = OrderManager(executor, StrategyParams())
    pos = om.evaluate_and_execute("XAUUSD", side=1, sl=2640.0, tp=2660.0, current_price=2650.0, spread_pips=1.8)

    assert pos is None
    assert any("ekuitas" in l["message"].lower() or "0" in l["message"] for l in om.logs)


def test_no_hardcoded_bigboz_paths():
    """Memastikan tidak ada path hardcoded d:\\BigBoz yang tersisa di seluruh file produksi."""
    found = []
    for root, dirs, files in os.walk(FINEX_DIR):
        if any(ignored in root for ignored in ["__pycache__", "scratch", "archive", "data"]):
            continue
        for f in files:
            if f.endswith(".py"):
                fp = Path(root) / f
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    for line_num, line in enumerate(fh, 1):
                        if "bigboz" in line.lower():
                            found.append(f"{fp.name}:{line_num} -> {line.strip()[:60]}")
    assert not found, f"Path BigBoz masih ditemukan:\n" + "\n".join(found)
