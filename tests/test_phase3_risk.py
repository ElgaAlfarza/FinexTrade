"""Unit tests for Phase 3: Risiko Terpadu dan Eksekusi Bersih.

Verifies:
1. Monotonicity of profit-locking (no regression at 4.90 vs 5.00).
2. Dynamic lot sizing based on account equity, risk_pct, and SL pips.
3. Persistence of daily_r_pnl in SQLite surviving restarts.
4. Defense-in-depth guard against REAL and Cent accounts in MT5Executor.open_trade.
"""
from __future__ import annotations

import os
import sys
import types
import sqlite3
import pytest

BOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "finex-bot"))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

from tests.mock_mt5 import MockMT5, MockAccountInfo
sys.modules["MetaTrader5"] = MockMT5()

from execution import PaperExecutor, MT5Executor, OrderManager, Position
from config import StrategyParams
from risk import lot_size, can_trade_today


def test_profit_lock_strictly_monotonic():
    """Verify profit lock is strictly non-decreasing as peak profit increases."""
    executor = PaperExecutor(initial_balance=10000.0)
    params = StrategyParams(active_technique="EMA_TREND")
    om = OrderManager(executor, params)

    pos = executor.open_trade("XAUUSD", "BUY", lots=0.01, sl=2640.0, tp=2660.0, risk_pips=100.0)
    assert pos is not None

    peaks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.90, 5.00, 5.50, 6.00, 10.00]
    locked_history = []

    for pk in peaks:
        pos.floating_pnl = pk
        om.manage_active_positions()
        locked = om._locked_profit_by_ticket.get(pos.ticket, 0.0)
        locked_history.append((pk, locked))

    # Verify no regression anywhere in the curve
    for i in range(1, len(locked_history)):
        prev_pk, prev_lk = locked_history[i - 1]
        cur_pk, cur_lk = locked_history[i]
        assert cur_lk >= prev_lk, f"Profit lock regressed from ${prev_lk} at peak ${prev_pk} to ${cur_lk} at peak ${cur_pk}!"

    # Specifically check peak 4.90 vs 5.00 (the specific bug reported in P1-4)
    lk_490 = [lk for pk, lk in locked_history if pk == 4.90][0]
    lk_500 = [lk for pk, lk in locked_history if pk == 5.00][0]
    assert lk_500 >= lk_490, f"Peak 5.00 locked ${lk_500} should be >= peak 4.90 locked ${lk_490}"


def test_dynamic_lot_sizing_calculation():
    """Verify dynamic position sizing scales correctly with equity and SL distance."""
    # Equity $1000, risk 1% ($10), SL 20 pips, pip_val $10/lot -> lot = 10 / (20 * 10) = 0.05
    calc_lot = lot_size(
        equity=1000.0,
        risk_pct=1.0,
        sl_pips=20.0,
        pip_value_per_lot=10.0,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=100.0,
    )
    assert calc_lot == 0.05

    # If equity is too small for min_lot risk:
    # Equity $50, risk 1% ($0.50), SL 50 pips, pip_val $10/lot -> raw = 0.50 / 500 = 0.001 < min_lot 0.01
    calc_lot_zero = lot_size(
        equity=50.0,
        risk_pct=1.0,
        sl_pips=50.0,
        pip_value_per_lot=10.0,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=100.0,
    )
    assert calc_lot_zero == 0.0

    # In execute_scalping: when scalping_lot is 0 (dynamic mode)
    executor = PaperExecutor(initial_balance=1000.0)
    params = StrategyParams(
        active_technique="EMA_TREND",
        scalping_lot=0.0,  # 0 indicates dynamic lot mode
        risk_pct=1.0,
    )
    om = OrderManager(executor, params)

    pos = om.execute_scalping(
        symbol="XAUUSD",
        side=1,
        sl=2645.0,
        tp=2665.0,
        current_price=2650.0,
        spread_pips=1.5,
        lot=0.0,  # Trigger dynamic sizing
    )
    assert pos is not None
    assert pos.lots > 0.0


def test_circuit_breaker_persistence_across_restart(tmp_path):
    """Verify daily_r_pnl survives bot restart via SQLite state storage."""
    db_file = str(tmp_path / "test_bot_state.db")
    executor = PaperExecutor(initial_balance=10000.0)
    params = StrategyParams(max_daily_loss_r=3.0)
    om1 = OrderManager(executor, params, db_path=db_file)

    # Register two -1.0R losses
    om1.risk_usd_by_ticket[101] = 50.0
    om1.register_closed_trade({"ticket": 101, "pnl": -50.0})
    om1.risk_usd_by_ticket[102] = 50.0
    om1.register_closed_trade({"ticket": 102, "pnl": -50.0})

    assert om1.daily_r_pnl == -2.0

    # Simulate bot crash & restart: instantiate a new OrderManager pointing to same DB
    om2 = OrderManager(executor, params, db_path=db_file)

    assert om2.daily_r_pnl == -2.0

    # Another -1.2R loss pushes it to -3.2R, tripping circuit breaker
    om2.risk_usd_by_ticket[103] = 50.0
    om2.register_closed_trade({"ticket": 103, "pnl": -60.0})

    assert om2.daily_r_pnl == -3.2
    assert not can_trade_today(om2.daily_r_pnl, params.max_daily_loss_r)

    # Attempting to open trade must be rejected by circuit breaker
    rejected = om2.execute_scalping("XAUUSD", 1, 2640.0, 2660.0, 2650.0, 1.5)
    assert rejected is None


def test_mt5_executor_real_and_cent_guard(monkeypatch):
    """Verify MT5Executor.open_trade rejects REAL account and Cent account."""
    mock = MockMT5()
    session = types.SimpleNamespace(mt5=mock)
    executor = MT5Executor(session)

    # 1. Test REAL account without ALLOW_LIVE_REAL=1
    monkeypatch.setenv("ALLOW_LIVE_REAL", "0")
    mock._account.trade_mode = 2  # ACCOUNT_TRADE_MODE_REAL
    mock._account.currency = "USD"

    res_real = executor.open_trade("XAUUSD", "BUY", lots=0.01, sl=2640.0, tp=2660.0, risk_pips=50.0)
    assert res_real is None
    assert "REAL diblokir" in executor.last_order_error

    # 2. Test Cent account (currency == CNT)
    mock._account.trade_mode = 0  # DEMO
    mock._account.currency = "CNT"

    res_cent = executor.open_trade("XAUUSD", "BUY", lots=0.01, sl=2640.0, tp=2660.0, risk_pips=50.0)
    assert res_cent is None
    assert "Cent" in executor.last_order_error
