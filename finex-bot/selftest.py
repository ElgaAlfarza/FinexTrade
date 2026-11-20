"""Self-test komprehensif: membuktikan seluruh arsitektur bot bekerja sesuai aturan.

Yang diuji:
1. Ukuran lot (risk.py)
2. Mekanika backtest, SL, TP, Break-Even, dan biaya spread/slippage (backtest.py)
3. Determinisme dan tidak adanya lookahead pada sinyal (strategy.py)
4. Paper Trading Engine & Order Manager (execution.py)
5. End-to-end backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import compute_stats, format_report, run_backtest, with_params
from config import PRESETS, StrategyParams
from execution import OrderManager, PaperExecutor
from risk import can_trade_today, lot_size
from strategy import generate_signals


def make_synthetic(n=60000, seed=7, freq="5min", base=1.10):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq=freq)
    regime = np.repeat(rng.choice([-1, 0, 1], size=n // 500 + 1, p=[0.3, 0.4, 0.3]), 500)[:n]
    vol = (0.00012 * np.exp(0.3 * rng.standard_normal(n // 200 + 1))).repeat(200)[:n]
    ret = regime * 0.00002 + vol * rng.standard_normal(n)
    close = base * np.exp(np.cumsum(ret))
    open_ = np.r_[base, close[:-1]]
    wu = np.abs(rng.standard_normal(n)) * vol * base * 0.6
    wd = np.abs(rng.standard_normal(n)) * vol * base * 0.6
    return pd.DataFrame(
        dict(open=open_, high=np.maximum(open_, close) + wu, low=np.minimum(open_, close) - wd, close=close),
        index=idx,
    )


def flat_frame(n=20, price=1.1000):
    idx = pd.date_range("2025-03-03", periods=n, freq="5min")
    return pd.DataFrame(dict(open=price, high=price, low=price, close=price), index=idx)


def injected(df, bar, side, sl, tp):
    sig = pd.DataFrame(dict(side=0, sl=np.nan, tp=np.nan), index=df.index)
    sig.loc[df.index[bar], ["side", "sl", "tp"]] = [side, sl, tp]
    return sig


def test_lot_size():
    assert lot_size(10000, 1, 20, 10) == 0.5
    assert lot_size(10000, 1, 33, 10) == 0.30          # dibulatkan ke bawah
    assert lot_size(100, 1, 50, 10) == 0.0             # di bawah lot minimum
    print("OK  lot_size")


def test_mechanics():
    p = with_params(StrategyParams(), spread_pips=0, slippage_pips=0, cooldown_bars=0, use_break_even=False)
    cases = [
        ("BUY kena TP", 1, 1.0990, 1.1020, dict(high=1.1021), +2.0, "TP"),
        ("BUY kena SL", 1, 1.0990, 1.1020, dict(low=1.0989), -1.0, "SL"),
        ("BUY SL dan TP satu candle -> SL dulu", 1, 1.0990, 1.1020, dict(high=1.1021, low=1.0989), -1.0, "SL"),
        ("SELL kena TP", -1, 1.1010, 1.0980, dict(low=1.0979), +2.0, "TP"),
        ("SELL kena SL", -1, 1.1010, 1.0980, dict(high=1.1011), -1.0, "SL"),
    ]
    for name, side, sl, tp, bar6, want_r, want_reason in cases:
        df = flat_frame()
        for col, val in bar6.items():
            df.iloc[6, df.columns.get_loc(col)] = val
        t = run_backtest(df, p, signals=injected(df, 2, side, sl, tp))
        assert len(t) == 1, f"{name}: harus 1 trade, dapat {len(t)}"
        assert t.loc[0, "reason"] == want_reason, f"{name}: alasan {t.loc[0, 'reason']}"
        assert abs(t.loc[0, "r"] - want_r) < 1e-6, f"{name}: r={t.loc[0, 'r']}"
    # biaya spread membuat hasil BUY lebih buruk dari tanpa biaya
    df = flat_frame()
    df.iloc[6, df.columns.get_loc("high")] = 1.1021
    cost = with_params(p, spread_pips=1.0, slippage_pips=0.5)
    t = run_backtest(df, cost, signals=injected(df, 2, 1, 1.0990, 1.1020))
    assert 0 < t.loc[0, "r"] < 2.0

    # Test Break-Even logic
    p_be = with_params(p, use_break_even=True, break_even_r=1.0)
    df_be = flat_frame(n=30)
    # Candle 6 menyentuh +1R (1.1010) lalu turun ke level entry (1.1000)
    df_be.iloc[6, df_be.columns.get_loc("high")] = 1.1011 # trigger BE
    df_be.iloc[8, df_be.columns.get_loc("low")] = 1.0995  # tembus BE
    t_be = run_backtest(df_be, p_be, signals=injected(df_be, 2, 1, 1.0990, 1.1020))
    assert len(t_be) == 1
    assert t_be.loc[0, "reason"] == "BE"
    assert t_be.loc[0, "r"] >= 0  # tidak rugi -1R

    print("OK  mekanika backtest (TP, SL, SL dulu, sell, biaya, Break-Even)")


def test_no_lookahead():
    df = make_synthetic()
    p = with_params(StrategyParams(), **PRESETS["M5"])
    full = generate_signals(df, p)
    for cut in (31234, 45678):
        part = generate_signals(df.iloc[:cut], p)
        pd.testing.assert_frame_equal(full.iloc[:cut], part, check_freq=False)
    n_sig = int((full["side"] != 0).sum())
    assert n_sig > 0, "tidak ada sinyal sama sekali pada data uji"
    print(f"OK  tanpa lookahead (sinyal identik saat data dipotong; {n_sig} sinyal mentah)")


def test_ma_filter():
    from strategy import ma_direction, ma_htf_frame, moving_average
    # 1) nilai SMA/EMA benar pada contoh kecil
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert abs(moving_average(s, 3, "SMA").iloc[-1] - 4.0) < 1e-9
    assert abs(moving_average(s, 3, "EMA").iloc[-1] - 4.0625) < 1e-9  # alpha=0.5: 1, 1.5, 2.25, 3.125, 4.0625
    assert moving_average(s, 3, "SMA").iloc[:2].isna().all()
    # 2) arah MA
    import numpy as _np
    frame = pd.DataFrame({"htf_close": [10, 8, 10], "ma_fast": [9, 9, _np.nan], "ma_slow": [9.5, 9.5, 9.5]})
    assert list(ma_direction(frame, "price_slow")) == [1, -1, 1]
    assert list(ma_direction(frame, "cross"))[:2] == [-1, -1]
    # 3) filter hanya MENGURANGI sinyal, tidak menambah, dan tanpa lookahead
    df = make_synthetic()
    base = with_params(StrategyParams(), **PRESETS["M5"], use_ma_filter=False)
    for mode in ("price_slow", "cross", "price_both"):
        flt = with_params(base, use_ma_filter=True, ma_mode=mode)
        a, b = generate_signals(df, base), generate_signals(df, flt)
        assert ((b["side"] != 0) <= (a["side"] != 0)).all(), f"{mode}: filter menambah sinyal"
        assert (b["side"][b["side"] != 0] == a["side"][b["side"] != 0]).all()
        for cut in (31234, 45678):
            part = generate_signals(df.iloc[:cut], flt)
            pd.testing.assert_frame_equal(b.iloc[:cut], part, check_freq=False)
        print(f"OK  filter MA mode {mode}: {int((a['side']!=0).sum())} -> {int((b['side']!=0).sum())} sinyal, tanpa lookahead")


def test_explore_and_circuit_breaker():
    """Mode latihan: batas posisi, laju entry, circuit breaker benar-benar menghitung R, dan guard akun REAL."""
    import logging
    logging.disable(logging.CRITICAL)
    from live_engine import LiveBotEngine
    from generate_sample_data import generate_market_data
    raw = generate_market_data("XAUUSD", "M1", 3000, seed=11)
    raw["time"] = pd.to_datetime(raw["time"])
    dfa = raw.set_index("time")[["open", "high", "low", "close"]]

    def drive(eng, n=600, start=400):
        eng._market_closed = lambda: False
        for i in range(start, start + n):
            df = dfa.iloc[: i + 1]
            bar = df.iloc[-1]
            sp = eng.params.spread_pips * eng.params.pip_size
            for px in (bar["open"], bar["low"] if bar["close"] < bar["open"] else bar["high"], bar["close"]):
                eng.executor.update_prices("XAUUSD", float(px), float(px) + sp)
            eng.current_bid, eng.current_ask = float(bar["close"]), float(bar["close"]) + sp
            eng.df_candles = df
            eng._on_new_candle_closed(bar, df)
            assert len(eng.executor.get_positions()) <= eng.params.explore_max_open + 0
        return eng.executor.get_trade_history()

    # mati secara default
    e = LiveBotEngine("XAUUSD", "M1", use_paper=True)
    assert e.params.explore_mode is False
    # aktif: ada entry, batas posisi terjaga, R harian tercatat
    e = LiveBotEngine("XAUUSD", "M1", use_paper=True)
    e.set_explore(True, every_bars=3, max_open=2, daily_loss_r=50.0)
    hist = drive(e)
    assert len(hist) > 20, f"mode latihan terlalu sedikit entry: {len(hist)}"
    assert abs(e.order_manager.daily_r_pnl - sum(t["r"] for t in hist if t["ticket"] not in e.order_manager.risk_usd_by_ticket)) < 1e-6
    # setelan bertahan saat ganti market
    e.set_market("XAUUSD", "M5")
    assert e.params.explore_mode is True and e.params.explore_max_open == 2
    # circuit breaker: R harian dihitung dari trade tutup dan entry baru ditolak saat batas tercapai
    e = LiveBotEngine("XAUUSD", "M1", use_paper=True)
    e.set_explore(True, every_bars=1, max_open=3, daily_loss_r=1.5)
    om = e.order_manager
    e.executor.update_prices("XAUUSD", 2700.0, 2700.25)
    for tk in (9001, 9002):
        om.risk_usd_by_ticket[tk] = 100.0
        om.register_closed_trade({"ticket": tk, "pnl": -100.0})      # tanpa field r (jalur MT5)
    assert abs(om.daily_r_pnl + 2.0) < 1e-9, om.daily_r_pnl
    assert om.evaluate_and_execute("XAUUSD", 1, 2698.5, 2704.0, 2700.25, 2.5) is None
    assert len(e.executor.get_positions()) == 0
    # sebelum batas: entry yang valid lolos
    om.daily_r_pnl = -1.0
    assert om.evaluate_and_execute("XAUUSD", 1, 2698.5, 2704.0, 2700.25, 2.5) is not None
    # mode normal memakai batas max_daily_loss_r (3R), bukan batas latihan
    e = LiveBotEngine("XAUUSD", "M5", use_paper=True)
    e.executor.update_prices("XAUUSD", 2700.0, 2700.25)
    e.order_manager.daily_r_pnl = -3.0
    assert e.order_manager.evaluate_and_execute("XAUUSD", 1, 2697.5, 2704.0, 2700.25, 2.5) is None
    # akun REAL ditolak
    e = LiveBotEngine("XAUUSD", "M1", use_paper=True)
    e.get_mode = lambda: "MT5_REAL"
    try:
        e.set_explore(True)
        raise AssertionError("akun REAL seharusnya ditolak")
    except ValueError:
        pass
    print("OK  mode latihan (batas posisi, laju entry, circuit breaker menghitung R, guard akun REAL)")


def test_execution_and_paper():
    # Uji simulator paper trading
    executor = PaperExecutor(initial_balance=10000.0)
    params = StrategyParams()
    manager = OrderManager(executor, params)

    # Cek circuit breaker
    assert can_trade_today(0.0, 3.0) is True
    assert can_trade_today(-2.5, 3.0) is True
    assert can_trade_today(-3.1, 3.0) is False

    # Set harga awal
    executor.update_prices("EURUSD", 1.0850, 1.0851)

    # Buka order buy
    pos = executor.open_trade("EURUSD", "BUY", 0.5, 1.0830, 1.0890, 20.0, "TestBuy")
    assert pos is not None
    assert len(executor.get_positions()) == 1

    # Simulasi kenaikan harga -> TP
    executor.update_prices("EURUSD", 1.0891, 1.0892)
    assert len(executor.get_positions()) == 0
    assert len(executor.trade_history) == 1
    assert executor.trade_history[0]["reason"] == "TP"
    assert executor.balance > 10000.0

    print("OK  engine eksekusi & paper trading (order, SL/TP trigger, circuit breaker)")


def test_end_to_end():
    df = make_synthetic()
    for sl_mode in ("structure", "candle"):
        p = with_params(StrategyParams(), sl_mode=sl_mode, **PRESETS["M5"])
        trades = run_backtest(df, p)
        if len(trades):
            tp = trades[trades["reason"] == "TP"]
            sl = trades[trades["reason"] == "SL"]
            assert (tp["r"] >= p.min_rr - 1e-6).all(), "trade TP harus >= RR minimum"
            assert (sl["r"] <= -0.999).all(), "trade SL harus sekitar -1R atau lebih buruk"
        print(f"\n[data sintetis, SL {sl_mode}] hasil tidak bermakna untuk profit, hanya cek kode jalan:")
        print(format_report(compute_stats(trades, p), p))
    print("\nOK  end-to-end")


def test_forecast_and_learner():
    """Uji kejujuran statistik forecast (tanpa lookahead, anti-tebak acak) & normalisasi Learner."""
    import os
    import tempfile
    from config import build_params
    from forecast import _features, forecast
    from learner import Learner

    # 1) build_params tidak menimpa batas SL Emas dengan angka EURUSD
    px_m5 = build_params("XAUUSD", "M5")
    px_m15 = build_params("XAUUSD", "M15")
    assert px_m5.max_sl_pips >= 50.0 and px_m15.max_sl_pips > px_m5.max_sl_pips
    assert px_m5.max_spread_pips >= 5.0

    # 2) Ekstraksi fitur 100% kausal (tanpa lookahead)
    df_syn = make_synthetic(n=1500, seed=21)
    X_full = _features(df_syn)
    X_cut = _features(df_syn.iloc[:900])
    assert np.allclose(X_full[:900], X_cut, equal_nan=True, atol=1e-10)

    # 3) Honesty Gate: pada Random Walk murni, bot WAJIB menolak menebak (NETRAL & reliable=False)
    rng = np.random.default_rng(12345)
    n_rw = 1600
    idx_rw = pd.date_range("2026-01-01", periods=n_rw, freq="5min")
    ret_rw = 0.0004 * rng.standard_normal(n_rw)
    c_rw = 1.1000 * np.exp(np.cumsum(ret_rw))
    o_rw = np.r_[1.1000, c_rw[:-1]]
    sp_rw = np.abs(rng.standard_normal(n_rw)) * 0.0002
    df_rw = pd.DataFrame({"open": o_rw, "high": np.maximum(o_rw, c_rw) + sp_rw, "low": np.minimum(o_rw, c_rw) - sp_rw, "close": c_rw}, index=idx_rw)
    fc_rw = forecast(df_rw, h=5, pip=0.0001, timeframe="M5", symbol="EURUSD")
    assert fc_rw["ok"] is True
    assert fc_rw["reliable"] is False and fc_rw["direction"] == "NETRAL", f"Random walk harus NETRAL: {fc_rw}"
    assert len(fc_rw["steps"]) == 5
    assert fc_rw["steps"][0]["sigma_pips"] < fc_rw["steps"][-1]["sigma_pips"]

    # 4) Deteksi sinyal kausal nyata: saat memang ada autokorelasi kuat, Walk-Forward menangkapnya
    rng2 = np.random.default_rng(99)
    ret_ar = np.zeros(n_rw)
    eps = 0.0003 * rng2.standard_normal(n_rw)
    for t in range(1, n_rw):
        ret_ar[t] = 0.55 * ret_ar[t - 1] + eps[t]
    ret_ar[-4:] = 0.0012  # impuls positif kuat di akhir
    c_ar = 1.1000 * np.exp(np.cumsum(ret_ar))
    o_ar = np.r_[1.1000, c_ar[:-1]]
    df_ar = pd.DataFrame({"open": o_ar, "high": np.maximum(o_ar, c_ar) + 0.0001, "low": np.minimum(o_ar, c_ar) - 0.0001, "close": c_ar}, index=idx_rw)
    fc_ar = forecast(df_ar, h=5, pip=0.0001, timeframe="M5", symbol="EURUSD")
    assert fc_ar["ok"] is True and fc_ar["reliable"] is True and fc_ar["direction"] == "NAIK", f"Sinyal AR harus NAIK: {fc_ar}"

    # 5) Learner online dengan Welford Z-score tahan terhadap fitur skala besar (spread=35 vs wick=0.4)
    tmp_state = os.path.join(tempfile.gettempdir(), "test_finex_learner_norm.json")
    if os.path.exists(tmp_state):
        os.remove(tmp_state)
    lrn = Learner(path=tmp_state)
    for i in range(40):
        good = (i % 2 == 0)
        feats = {
            "trend": 1.0,
            "zone_depth": 0.62 if good else 0.51,
            "wick": 0.65 if good else 0.20,
            "rr": 2.0,
            "sl_atr": 1.2,
            "hour_sin": 0.5,
            "hour_cos": -0.5,
            "spread": 3.5 if good else 35.0,
        }
        lrn.update(feats, good)
    p_good = lrn.predict({"trend": 1.0, "zone_depth": 0.62, "wick": 0.65, "rr": 2.0, "sl_atr": 1.2, "hour_sin": 0.5, "hour_cos": -0.5, "spread": 3.5})
    p_bad = lrn.predict({"trend": 1.0, "zone_depth": 0.51, "wick": 0.20, "rr": 2.0, "sl_atr": 1.2, "hour_sin": 0.5, "hour_cos": -0.5, "spread": 35.0})
    if os.path.exists(tmp_state):
        os.remove(tmp_state)
    assert p_good > p_bad + 0.15, f"Learner gagal membedakan setup bagus vs spread lebar: {p_good} vs {p_bad}"
    print("OK  forecast walk-forward (tanpa lookahead, anti-tebak acak, kerucut EWMA) & learner Welford Z-score")


if __name__ == "__main__":
    test_lot_size()
    test_mechanics()
    test_no_lookahead()
    test_ma_filter()
    test_explore_and_circuit_breaker()
    test_execution_and_paper()
    test_forecast_and_learner()
    test_end_to_end()
    print("\nSEMUA TEST LULUS (Tahap 1, 2, 3, Forecast Walk-Forward & Learner Terverifikasi!)")
