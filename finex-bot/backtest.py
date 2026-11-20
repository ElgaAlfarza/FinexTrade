"""Mesin backtest dan statistik.

Asumsi eksekusi (sengaja konservatif):
- Sinyal dihitung saat candle i tutup, order masuk di open candle i+1.
- Data MT5 berisi harga BID. Buy masuk di ask (open + spread + slippage), sell masuk di bid.
- Jika SL dan TP tersentuh di candle yang sama, SL dianggap kena lebih dulu.
- Gap melewati SL diisi di harga open, bukan di harga SL.
- Satu posisi pada satu waktu. Ada cooldown dan batas rugi harian.
- Break-Even (opsional): geser SL ke entry saat floating profit mencapai target R.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from config import StrategyParams
from strategy import generate_signals


def run_backtest(df: pd.DataFrame, p: StrategyParams, signals: pd.DataFrame | None = None) -> pd.DataFrame:
    sig = generate_signals(df, p) if signals is None else signals
    side = sig["side"].to_numpy()
    slv = sig["sl"].to_numpy(float)
    tpv = sig["tp"].to_numpy(float)
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    idx = df.index
    n = len(df)
    pip = p.pip_size
    spread = p.spread_pips * pip
    slip = p.slippage_pips * pip

    trades = []
    daily: dict = {}
    i = 0
    while i < n - 1:
        s = int(side[i])
        if s == 0:
            i += 1
            continue
        e = i + 1
        day = idx[e].date()
        if daily.get(day, 0.0) <= -p.max_daily_loss_r:
            i += 1
            continue

        sl, tp = slv[i], tpv[i]
        entry = o[e] + spread + slip if s == 1 else o[e] - slip
        risk = (entry - sl) if s == 1 else (sl - entry)
        reward = (tp - entry) if s == 1 else (entry - tp)
        if not (risk > 0 and reward > 0):
            i += 1
            continue
        risk_pips = risk / pip
        if risk_pips > p.max_sl_pips or risk_pips < p.min_sl_pips or reward / risk < p.min_rr:
            i += 1
            continue

        last = min(n - 1, e + p.max_bars_in_trade - 1)
        exit_price, reason, j = None, "TIMEOUT", last
        active_sl = sl
        be_active = False

        for j in range(e, last + 1):
            if s == 1:
                # Cek proteksi Break Even jika aktif
                if getattr(p, "use_break_even", False) and not be_active:
                    be_trigger = entry + p.break_even_r * risk
                    if h[j] >= be_trigger:
                        active_sl = max(active_sl, entry + p.be_buffer_pips * pip)
                        be_active = True

                if l[j] <= active_sl:
                    exit_price, reason = min(active_sl, o[j]) - slip, ("BE" if be_active else "SL")
                    break
                if h[j] >= tp:
                    exit_price, reason = max(tp, o[j]), "TP"
                    break
            else:
                # Cek proteksi Break Even untuk sell
                if getattr(p, "use_break_even", False) and not be_active:
                    be_trigger = entry - p.break_even_r * risk
                    if l[j] + spread <= be_trigger:
                        active_sl = min(active_sl, entry - p.be_buffer_pips * pip)
                        be_active = True

                if h[j] + spread >= active_sl:
                    exit_price, reason = max(active_sl, o[j] + spread) + slip, ("BE" if be_active else "SL")
                    break
                if l[j] + spread <= tp:
                    exit_price, reason = min(tp, o[j] + spread), "TP"
                    break

        if exit_price is None:
            exit_price = c[last] if s == 1 else c[last] + spread

        pnl = (exit_price - entry) if s == 1 else (entry - exit_price)
        r = pnl / risk
        daily[day] = daily.get(day, 0.0) + r
        trades.append(
            dict(
                entry_time=idx[e], exit_time=idx[j], side="BUY" if s == 1 else "SELL",
                entry=entry, sl=sl, tp=tp, exit=exit_price, reason=reason,
                r=r, risk_pips=risk_pips, bars=j - e + 1,
            )
        )
        i = j + 1 + p.cooldown_bars

    return pd.DataFrame(trades)


def wilson_ci(wins: int, n: int, z: float = 1.96):
    """Interval kepercayaan 95% untuk win rate (Wilson)."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = wins / n
    denom = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / denom
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def compute_stats(trades: pd.DataFrame, p: StrategyParams, start_equity: float = 10000.0) -> dict:
    if trades.empty:
        return {"trades": 0, "equity_curve": [{"time": "Start", "equity": start_equity}]}
    r = trades["r"].to_numpy()
    wins = r > 0
    n = len(r)
    gross_win = float(r[wins].sum())
    gross_loss = float(-r[~wins].sum())
    avg_win = float(r[wins].mean()) if wins.any() else 0.0
    avg_loss = float(r[~wins].mean()) if (~wins).any() else 0.0

    eq = start_equity * np.cumprod(1 + p.risk_pct / 100.0 * r)
    peak = np.maximum.accumulate(np.concatenate([[start_equity], eq]))[1:]
    dd_series = eq / peak - 1
    max_dd = float(dd_series.min())

    streak = best = 0
    for w in wins:
        streak = 0 if w else streak + 1
        best = max(best, streak)

    lo, hi = wilson_ci(int(wins.sum()), n)
    be_wr = abs(avg_loss) / (avg_win + abs(avg_loss)) if (avg_win + abs(avg_loss)) > 0 else float("nan")

    # Curve points untuk visualisasi dashboard
    equity_curve = [{"idx": 0, "time": str(trades["entry_time"].iloc[0]), "equity": round(start_equity, 2), "dd": 0.0}]
    for idx_t, (t_time, eq_val, dd_val) in enumerate(zip(trades["exit_time"], eq, dd_series)):
        equity_curve.append({
            "idx": idx_t + 1,
            "time": str(t_time),
            "equity": round(float(eq_val), 2),
            "dd": round(float(dd_val * 100), 2)
        })

    return dict(
        trades=n,
        win_rate=float(wins.mean()),
        win_rate_ci95=(lo, hi),
        avg_win_r=avg_win,
        avg_loss_r=avg_loss,
        breakeven_win_rate=be_wr,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        expectancy_r=float(r.mean()),
        total_r=float(r.sum()),
        max_drawdown=max_dd,
        end_equity=float(eq[-1]),
        start_equity=start_equity,
        max_consecutive_losses=int(best),
        tp_count=int((trades["reason"] == "TP").sum()),
        sl_count=int((trades["reason"] == "SL").sum()),
        be_count=int((trades["reason"] == "BE").sum()),
        timeout_count=int((trades["reason"] == "TIMEOUT").sum()),
        equity_curve=equity_curve,
    )


def format_report(stats: dict, p: StrategyParams, target_wr: float = 0.90) -> str:
    if stats.get("trades", 0) == 0:
        return "Tidak ada trade yang terbentuk. Longgarkan filter atau gunakan data yang lebih panjang."
    lo, hi = stats["win_rate_ci95"]
    be_str = f", BE {stats['be_count']}" if stats.get("be_count", 0) > 0 else ""
    lines = [
        f"Jumlah trade        : {stats['trades']}  (TP {stats['tp_count']}, SL {stats['sl_count']}{be_str}, timeout {stats['timeout_count']})",
        f"Win rate            : {stats['win_rate']*100:.1f}%  (interval 95%: {lo*100:.1f}% sampai {hi*100:.1f}%)",
        f"Rata-rata menang    : {stats['avg_win_r']:+.2f} R   | rata-rata kalah: {stats['avg_loss_r']:+.2f} R",
        f"Win rate impas      : {stats['breakeven_win_rate']*100:.1f}%  (di bawah ini rugi pada rasio saat ini)",
        f"Profit factor       : {stats['profit_factor']:.2f}",
        f"Expectancy          : {stats['expectancy_r']:+.3f} R per trade   | total {stats['total_r']:+.1f} R",
        f"Max drawdown        : {stats['max_drawdown']*100:.1f}%  (risiko {p.risk_pct}% per trade)",
        f"Ekuitas akhir       : {stats['end_equity']:,.0f} dari {stats['start_equity']:,.0f}",
        f"Kalah beruntun maks : {stats['max_consecutive_losses']}",
    ]
    if stats["trades"] < 100:
        lines.append("Catatan: sampel kurang dari 100 trade, angka di atas belum bisa dipercaya.")
    if hi < target_wr:
        lines.append(f"Target win rate {target_wr*100:.0f}% TIDAK tercapai: batas atas interval 95% hanya {hi*100:.1f}%.")
    elif lo >= target_wr:
        lines.append(f"Target win rate {target_wr*100:.0f}% tercapai pada sampel ini. Ulangi di data lain dan di demo sebelum dipercaya.")
    else:
        lines.append(f"Target win rate {target_wr*100:.0f}% belum bisa dipastikan: sampel terlalu kecil.")
    if stats["expectancy_r"] <= 0:
        lines.append("Expectancy tidak positif: strategi ini rugi setelah spread dan slippage.")
    return "\n".join(lines)


def with_params(p: StrategyParams, **kw) -> StrategyParams:
    return dataclasses.replace(p, **kw)


def run_walk_forward_backtest(
    df: pd.DataFrame,
    p: StrategyParams,
    train_ratio: float = 0.7,
) -> dict:
    """Jalankan evaluasi split In-Sample (Train) vs Out-of-Sample (Test) secara jujur tanpa curve-fitting."""
    n = len(df)
    split_idx = int(n * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_trades = run_backtest(train_df, p)
    test_trades = run_backtest(test_df, p)

    train_stats = compute_stats(train_trades, p)
    test_stats = compute_stats(test_trades, p)

    return {
        "train_range": (str(train_df.index[0]), str(train_df.index[-1])),
        "test_range": (str(test_df.index[0]), str(test_df.index[-1])),
        "train_stats": train_stats,
        "test_stats": test_stats,
        "train_report": format_report(train_stats, p),
        "test_report": format_report(test_stats, p),
    }
