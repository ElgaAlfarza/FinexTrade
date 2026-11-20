"""Tests for Phase 2: Jujurkan Sistem & Hapus Halusinasi / Gate Palsu.

Verifies:
1. Gemini analyst does not have fake auto-approval bypass and returns honest fallback.
2. Gemini news sentiment returns TIDAK TERSEDIA when offline (no fake bullish 78%).
3. CPI predictor defaults to sniper_enabled=False and unverified data returns action=WAIT.
4. live_engine._compute_active_signal computes deterministic scores (0.0 on WAIT) without Gemini modification or literal accuracy claims.
5. news_radar fail-closed behavior when calendar cache is older than 24h.
6. execution.py emergency cut-loss protects ALL positions without macro sniper exemption.
"""
from __future__ import annotations

import os
import sys
import time
import pytest
import pandas as pd
import numpy as np

# Ensure finex-bot is on sys.path
BOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "finex-bot"))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

from tests.mock_mt5 import MockMT5, MockAccountInfo
sys.modules["MetaTrader5"] = MockMT5()

from gemini_analyst import GeminiMarketAnalyst
from cpi_predictor import CpiPredictor
from news_radar import EconomicNewsRadar
from execution import PaperExecutor, OrderManager, Position
from config import StrategyParams


def test_gemini_no_fake_bypass():
    """Verify analyze_trade_setup does not have fake auto-approval for BUY/SELL."""
    analyst = GeminiMarketAnalyst(api_key="")
    assert not analyst.is_available()

    # Pre-trade setup evaluation when Gemini is not available
    res = analyst.analyze_trade_setup(
        symbol="XAUUSD",
        timeframe="M1",
        action="BUY",
        current_price=2600.0,
        proposed_tp=2605.0,
        proposed_sl=2595.0,
        target_usd=5.0,
        risk_usd=5.0,
    )
    # Must NOT return fake 88% confidence
    assert res.get("confidence") != 88
    assert res.get("confidence") is None
    assert res.get("gemini_active") is False
    assert res.get("decision") == "PASSTHROUGH"


def test_gemini_news_sentiment_honest_fallback():
    """Verify get_xauusd_news_sentiment does not return fake BULLISH 78% fallback."""
    analyst = GeminiMarketAnalyst(api_key="")
    res = analyst.get_xauusd_news_sentiment(force_refresh=True)

    assert res.get("active") is False
    assert res.get("sentiment") == "NETRAL"
    assert res.get("headline") == "TIDAK TERSEDIA"
    assert res.get("bias_confidence") == 0


def test_cpi_sniper_disabled_by_default_and_unverified():
    """Verify CpiPredictor defaults to disabled and unverified data refuses execution."""
    cpi = CpiPredictor()
    assert cpi.sniper_enabled is False

    eval_res = cpi.evaluate_cpi_auto_sniper(
        current_price=2650.0,
        pip_size=0.1,
        symbol="XAUUSD",
    )
    assert eval_res.get("should_execute") is False
    assert "nonaktif" in eval_res.get("reason", "").lower()

    # Even if enabled manually, unverified data must return WAIT
    cpi.sniper_enabled = True
    pred = cpi.predict_market_direction_by_data(symbol="XAUUSD")
    assert pred.get("action") == "WAIT"
    assert pred.get("confidence") == "TIDAK TERSEDIA"
    assert pred.get("probability_pct") == 0.0


def test_news_radar_fail_closed_stale_cache(monkeypatch):
    """Verify news radar activates shield when cache is stale (>24h) and network fetch fails."""
    import urllib.request

    def mock_urlopen(*args, **kwargs):
        raise ConnectionError("Network offline simulated")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    radar = EconomicNewsRadar()
    # Simulate stale cache older than 24 hours
    radar._last_fetch_ts = time.time() - 90000  # ~25 hours ago
    radar._cached_events = [{"title": "Old Event", "impact": "High", "date": "2025-01-01T00:00:00Z"}]

    status = radar.get_news_shield_status()
    assert status.get("is_shield_active") is True
    assert status.get("status") == "KALENDER_KEDALUWARSA_ENTRY_DITAHAN"
    assert "kedaluwarsa" in status.get("warning", "").lower()


def test_execution_emergency_cut_loss_protects_macro_sniper():
    """Verify OrderManager cuts loss even on macro sniper positions."""
    executor = PaperExecutor(initial_balance=1000.0)
    params = StrategyParams(
        active_technique="EMA_TREND",
        target_loss_usd=5.0,
    )
    om = OrderManager(executor, params)

    # Open a position marked as CPI_SNIPER
    pos = executor.open_trade(
        symbol="XAUUSD",
        side="BUY",
        lots=0.01,
        sl=2640.0,
        tp=2660.0,
        risk_pips=100.0,
        comment="CPI_SNIPER",
    )
    assert pos is not None
    pos.technique = "CPI_SNIPER"
    assert pos.ticket in [p.ticket for p in executor.get_positions()]

    # Simulate market crashing, floating PnL drops below -$5.00
    executor.update_prices("XAUUSD", bid=2643.0, ask=2643.5)
    current_pos = executor.get_positions()[0]
    assert current_pos.floating_pnl <= -5.0

    # Call manage_active_positions
    om.manage_active_positions()

    # Position must be CLOSED by cut-loss darurat (NOT exempted!)
    remaining = executor.get_positions()
    assert len(remaining) == 0


def test_live_engine_deterministic_signal():
    """Verify _compute_active_signal produces deterministic confidence and no fake news."""
    from live_engine import LiveBotEngine
    params = StrategyParams(active_technique="EMA_TREND")
    engine = LiveBotEngine(params=params, symbol="XAUUSD")

    # Generate 50 bars of synthetic flat data
    dates = pd.date_range("2026-01-01", periods=60, freq="1min")
    prices = [2650.0] * 60
    df = pd.DataFrame({
        "time": dates,
        "open": prices,
        "high": [p + 0.1 for p in prices],
        "low": [p - 0.1 for p in prices],
        "close": prices,
        "tick_volume": [100] * 60,
    })

    sig = engine._compute_active_signal(df, symbol="XAUUSD", bid=2650.0, ask=2650.2, closed_only=True)
    assert sig is not None
    # Flat market must be WAIT with 0.0 confidence
    assert sig["action"] == "WAIT"
    assert sig["confidence"] == 0.0
    assert sig["gemini_news"] is None
    # Must NOT contain literal 69.1%
    assert "69.1%" not in sig.get("reason", "")
