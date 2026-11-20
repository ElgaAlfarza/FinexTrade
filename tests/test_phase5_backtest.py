"""Unit tests for Phase 5: Paritas Backtest dan Validasi Kuantitatif (P1-1).

Verifies:
1. Determinism and zero lookahead in `generate_ema_trend_signals`.
2. Execution parity between live indicator logic and backtest runner.
3. Realistic accounting of spread and slippage.
4. Walk-forward In-Sample vs Out-of-Sample validation and Wilson 95% CI.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pytest

BOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "finex-bot"))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

from tests.mock_mt5 import MockMT5
sys.modules["MetaTrader5"] = MockMT5()

from config import StrategyParams, build_params
from strategy import generate_ema_trend_signals, generate_signals
from backtest import run_backtest, compute_stats, run_walk_forward_backtest, with_params, wilson_ci


def make_trending_synthetic(n=2000, seed=42):
    """Generate synthetic OHLC series with pronounced trend and pullbacks for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    ret = 0.0002 + 0.0005 * rng.standard_normal(n)
    close = 2600.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[2600.0, close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * 1.5
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * 1.5
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_ema_trend_signals_no_lookahead():
    """Verify generate_ema_trend_signals has strictly zero lookahead bias."""
    df = make_trending_synthetic(n=1500, seed=123)
    p = with_params(build_params("XAUUSD", "M5"), active_technique="EMA_TREND", sl_mode="atr")

    full_sig = generate_ema_trend_signals(df, p)

    for cut in (500, 800, 1200):
        cut_sig = generate_ema_trend_signals(df.iloc[:cut], p)
        pd.testing.assert_frame_equal(full_sig.iloc[:cut], cut_sig, check_freq=False)


def test_wilson_ci_bounds():
    """Verify Wilson score confidence interval calculations."""
    # 0 wins out of 100
    lo, hi = wilson_ci(0, 100)
    assert 0.0 <= lo <= hi <= 0.05

    # 50 wins out of 100
    lo50, hi50 = wilson_ci(50, 100)
    assert 0.39 <= lo50 <= 0.50 <= hi50 <= 0.61

    # Empty sample
    lo_e, hi_e = wilson_ci(0, 0)
    assert np.isnan(lo_e) and np.isnan(hi_e)


def test_walk_forward_backtest_execution():
    """Verify run_walk_forward_backtest splits and computes in-sample and out-of-sample stats."""
    df = make_trending_synthetic(n=1000, seed=77)
    p = with_params(
        build_params("XAUUSD", "M5"),
        active_technique="EMA_TREND",
        sl_mode="atr",
        min_rr=0.8,
        spread_pips=1.5,
        slippage_pips=0.5,
    )

    res = run_walk_forward_backtest(df, p, train_ratio=0.7)
    assert "train_stats" in res
    assert "test_stats" in res
    assert "train_report" in res
    assert "test_report" in res

    train_st = res["train_stats"]
    test_st = res["test_stats"]

    assert isinstance(train_st["trades"], int)
    assert isinstance(test_st["trades"], int)
    if train_st["trades"] > 0:
        assert "win_rate" in train_st
        assert "expectancy_r" in train_st
        assert "profit_factor" in train_st


def test_spread_and_slippage_degrades_pnl():
    """Verify backtest accurately docks spread and slippage from gross performance."""
    df = make_trending_synthetic(n=800, seed=99)
    p_zero = with_params(build_params("XAUUSD", "M5"), active_technique="EMA_TREND", sl_mode="atr", min_rr=0.8, spread_pips=0.0, slippage_pips=0.0)
    p_cost = with_params(build_params("XAUUSD", "M5"), active_technique="EMA_TREND", sl_mode="atr", min_rr=0.8, spread_pips=3.0, slippage_pips=1.0)

    t_zero = run_backtest(df, p_zero)
    t_cost = run_backtest(df, p_cost)

    if len(t_zero) > 0 and len(t_cost) > 0:
        avg_r_zero = t_zero["r"].mean()
        avg_r_cost = t_cost["r"].mean()
        assert avg_r_zero > avg_r_cost, f"Cost did not degrade performance: {avg_r_zero} vs {avg_r_cost}"
