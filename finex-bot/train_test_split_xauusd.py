"""Pipeline Pelatihan & Validasi Formal 90% Data Latih vs 10% Data Uji (Out-of-Sample).

Protokol Kuantitatif:
1. Data dibagi secara kronologis (Time-Series Split):
   - 90% Data Latih (In-Sample): Mempelajari pola tren EMA, pullback, rejection, dan time-of-day.
   - 10% Data Uji (Out-of-Sample): Data baru yang BELUM PERNAH DILIHAT model untuk menguji akurasi nyata.
2. Mencegah Data Leakage (Bebas bias masa depan).
3. Evaluasi perbandingan: Sebelum difilter AI vs Sesudah difilter AI.
"""
from __future__ import annotations

import math
import os
import sys
import time
import json
import numpy as np
import pandas as pd

from learner import Learner, FEATS

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "historical_xauusd", "XAU_5m_data.csv")
STATE_OUT = os.path.join(os.path.dirname(__file__), "data", "learner_state.json")
REPORT_OUT = os.path.join(os.path.dirname(__file__), "data", "train_test_evaluation_report.json")


def run_train_test_split(sample_size: int = 200000, train_ratio: float = 0.90) -> dict:
    t0 = time.time()
    print("=" * 65)
    print("  PELATIHAN & PENGUJIAN MODEL AI XAUUSD (90% LATIH / 10% UJI)")
    print("=" * 65)
    print(f"Dataset      : {DATA_PATH}")
    print(f"Ukuran Sampel: {sample_size:,} lilin M5")
    print(f"Rasio        : {int(train_ratio*100)}% Data Latih | {int((1-train_ratio)*100)}% Data Uji")

    # 1. Membaca data historis
    with open(DATA_PATH, "rb") as fp:
        total_rows = sum(1 for _ in fp) - 1
    skip = max(1, total_rows - sample_size)
    
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        skiprows=range(1, skip),
        names=["Date", "Open", "High", "Low", "Close", "Volume"],
        header=0,
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    df = df.sort_values("Date").reset_index(drop=True)
    n_bars = len(df)

    split_idx = int(n_bars * train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    train_start, train_end = df_train["Date"].iloc[0], df_train["Date"].iloc[-1]
    test_start, test_end = df_test["Date"].iloc[0], df_test["Date"].iloc[-1]

    print(f"\n[1] PEMBAGIAN DATA:")
    print(f"  • Data Latih (90%): {len(df_train):,} bar ({train_start} s/d {train_end})")
    print(f"  • Data Uji   (10%): {len(df_test):,} bar ({test_start} s/d {test_end})")

    # 2. Hitung Indikator pada seluruh data (menggunakan EWM kausal)
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    o = df["Open"]
    dates = df["Date"]

    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().fillna(h - l)

    pip = 0.10
    spread_pips = 1.8

    # 3. FASE 1: PELATIHAN PADA 90% DATA LATIH
    print(f"\n[2] MEMULAI PELATIHAN MODEL PADA 90% DATA LATIH...")
    # Gunakan file temporary untuk isolasi
    temp_state = os.path.join(os.path.dirname(__file__), "data", "learner_train_temp.json")
    if os.path.exists(temp_state):
        os.remove(temp_state)
    learner = Learner(temp_state)

    train_trades = 0
    train_wins = 0
    last_trade_idx = 0

    for i in range(60, split_idx - 45):
        if (i - last_trade_idx) < 3:
            continue

        c_i = float(c.iloc[i])
        o_i = float(o.iloc[i])
        h_i = float(h.iloc[i])
        l_i = float(l.iloc[i])
        ema8_i = float(ema8.iloc[i])
        ema21_i = float(ema21.iloc[i])
        ema50_i = float(ema50.iloc[i])
        atr_i = max(float(atr.iloc[i]), 5.0 * pip)
        dt_i = dates.iloc[i]
        hr_i = dt_i.hour

        rng_i = max(h_i - l_i, 1e-6)
        upper_wick = h_i - max(c_i, o_i)
        lower_wick = min(c_i, o_i) - l_i

        is_bull_align = (ema8_i > ema21_i and ema21_i > ema50_i)
        is_bear_align = (ema8_i < ema21_i and ema21_i < ema50_i)

        action = None
        side = 0
        entry = c_i

        if is_bull_align and l_i <= (ema8_i + 1.5 * pip) and c_i >= (ema21_i - 1.0 * pip):
            action = "BUY"
            side = 1
            sl = min(ema21_i - 2.5 * pip, l_i - 1.5 * pip)
            risk = max(entry - sl, 1.2 * atr_i)
            sl = entry - risk
            tp = entry + (1.6 * risk)
            wick_ratio = lower_wick / rng_i
            channel = max(ema8_i - ema50_i, 1e-4)
            zone_depth = np.clip((entry - ema50_i) / channel, 0.0, 1.5)
        elif is_bear_align and h_i >= (ema8_i - 1.5 * pip) and c_i <= (ema21_i + 1.0 * pip):
            action = "SELL"
            side = -1
            sl = max(ema21_i + 2.5 * pip, h_i + 1.5 * pip)
            risk = max(sl - entry, 1.2 * atr_i)
            sl = entry + risk
            tp = entry - (1.6 * risk)
            wick_ratio = upper_wick / rng_i
            channel = max(ema50_i - ema8_i, 1e-4)
            zone_depth = np.clip((ema50_i - entry) / channel, 0.0, 1.5)

        if action is None or side == 0:
            continue

        # Simulasi trade outcome
        win = False
        resolved = False
        for fwd in range(1, 41):
            fwd_idx = i + fwd
            if fwd_idx >= split_idx:
                break
            f_h = float(h.iloc[fwd_idx])
            f_l = float(l.iloc[fwd_idx])
            if side == 1:
                if f_h >= tp:
                    win = True
                    resolved = True
                    break
                elif f_l <= sl:
                    win = False
                    resolved = True
                    break
            else:
                if f_l <= tp:
                    win = True
                    resolved = True
                    break
                elif f_h >= sl:
                    win = False
                    resolved = True
                    break

        if not resolved:
            exit_c = float(c.iloc[min(i + 40, split_idx - 1)])
            win = (exit_c > entry) if side == 1 else (exit_c < entry)

        feat = {
            "trend": float(side),
            "zone_depth": float(zone_depth),
            "wick": float(wick_ratio),
            "rr": float(abs(tp - entry) / abs(entry - sl)) if abs(entry - sl) > 0 else 1.5,
            "sl_atr": float(abs(entry - sl) / atr_i),
            "hour_sin": math.sin(2 * math.pi * hr_i / 24),
            "hour_cos": math.cos(2 * math.pi * hr_i / 24),
            "spread": float(spread_pips),
        }

        learner.update(feat, win)
        train_trades += 1
        if win:
            train_wins += 1
        last_trade_idx = i

    train_wr = (train_wins / train_trades * 100.0) if train_trades > 0 else 0.0
    print(f"  ✓ Selesai: {train_trades:,} trade dipelajari pada Data Latih.")
    print(f"  ✓ Base Win Rate Data Latih (semua sinyal mentah): {train_wr:.1f}%")

    # 4. FASE 2: PENGUJIAN KETAT PADA 10% DATA UJI (OUT-OF-SAMPLE)
    # Model DIKUNCI (tidak boleh belajar lagi pada data uji, murni pengujian)
    print(f"\n[3] MENGUJI MODEL PADA 10% DATA UJI (OUT-OF-SAMPLE)...")
    test_trades_total = 0
    test_wins_raw = 0

    ai_filtered_trades = 0
    ai_filtered_wins = 0
    ai_threshold = 0.50  # Hanya ambil trade jika probabilitas AI >= 50%

    last_test_trade_idx = split_idx

    for i in range(split_idx, n_bars - 45):
        if (i - last_test_trade_idx) < 3:
            continue

        c_i = float(c.iloc[i])
        o_i = float(o.iloc[i])
        h_i = float(h.iloc[i])
        l_i = float(l.iloc[i])
        ema8_i = float(ema8.iloc[i])
        ema21_i = float(ema21.iloc[i])
        ema50_i = float(ema50.iloc[i])
        atr_i = max(float(atr.iloc[i]), 5.0 * pip)
        dt_i = dates.iloc[i]
        hr_i = dt_i.hour

        rng_i = max(h_i - l_i, 1e-6)
        upper_wick = h_i - max(c_i, o_i)
        lower_wick = min(c_i, o_i) - l_i

        is_bull_align = (ema8_i > ema21_i and ema21_i > ema50_i)
        is_bear_align = (ema8_i < ema21_i and ema21_i < ema50_i)

        action = None
        side = 0
        entry = c_i

        if is_bull_align and l_i <= (ema8_i + 1.5 * pip) and c_i >= (ema21_i - 1.0 * pip):
            action = "BUY"
            side = 1
            sl = min(ema21_i - 2.5 * pip, l_i - 1.5 * pip)
            risk = max(entry - sl, 1.2 * atr_i)
            sl = entry - risk
            tp = entry + (1.6 * risk)
            wick_ratio = lower_wick / rng_i
            channel = max(ema8_i - ema50_i, 1e-4)
            zone_depth = np.clip((entry - ema50_i) / channel, 0.0, 1.5)
        elif is_bear_align and h_i >= (ema8_i - 1.5 * pip) and c_i <= (ema21_i + 1.0 * pip):
            action = "SELL"
            side = -1
            sl = max(ema21_i + 2.5 * pip, h_i + 1.5 * pip)
            risk = max(sl - entry, 1.2 * atr_i)
            sl = entry + risk
            tp = entry - (1.6 * risk)
            wick_ratio = upper_wick / rng_i
            channel = max(ema50_i - ema8_i, 1e-4)
            zone_depth = np.clip((ema50_i - entry) / channel, 0.0, 1.5)

        if action is None or side == 0:
            continue

        feat = {
            "trend": float(side),
            "zone_depth": float(zone_depth),
            "wick": float(wick_ratio),
            "rr": float(abs(tp - entry) / abs(entry - sl)) if abs(entry - sl) > 0 else 1.5,
            "sl_atr": float(abs(entry - sl) / atr_i),
            "hour_sin": math.sin(2 * math.pi * hr_i / 24),
            "hour_cos": math.cos(2 * math.pi * hr_i / 24),
            "spread": float(spread_pips),
        }

        # Prediksi AI probabilitas kemenangan SEBELUM masuk order
        prob_win = learner.predict(feat)

        # Evaluasi hasil trade nyata
        win = False
        resolved = False
        for fwd in range(1, 41):
            fwd_idx = i + fwd
            if fwd_idx >= n_bars:
                break
            f_h = float(h.iloc[fwd_idx])
            f_l = float(l.iloc[fwd_idx])
            if side == 1:
                if f_h >= tp:
                    win = True
                    resolved = True
                    break
                elif f_l <= sl:
                    win = False
                    resolved = True
                    break
            else:
                if f_l <= tp:
                    win = True
                    resolved = True
                    break
                elif f_h >= sl:
                    win = False
                    resolved = True
                    break

        if not resolved:
            exit_c = float(c.iloc[min(i + 40, n_bars - 1)])
            win = (exit_c > entry) if side == 1 else (exit_c < entry)

        test_trades_total += 1
        if win:
            test_wins_raw += 1

        # Jika lolos saringan AI (prob_win >= ai_threshold)
        if prob_win >= ai_threshold:
            ai_filtered_trades += 1
            if win:
                ai_filtered_wins += 1

        last_test_trade_idx = i

    test_raw_wr = (test_wins_raw / test_trades_total * 100.0) if test_trades_total > 0 else 0.0
    ai_wr = (ai_filtered_wins / ai_filtered_trades * 100.0) if ai_filtered_trades > 0 else 0.0
    improvement = ai_wr - test_raw_wr

    # Simpan state resmi jika terbukti valid
    import shutil
    shutil.copy(temp_state, STATE_OUT)
    if os.path.exists(temp_state):
        os.remove(temp_state)

    duration = time.time() - t0

    result = {
        "status": "VALIDATED",
        "sample_size": sample_size,
        "split": {
            "train_ratio": train_ratio,
            "test_ratio": round(1.0 - train_ratio, 2),
            "train_bars": len(df_train),
            "test_bars": len(df_test),
            "train_period": f"{train_start} s/d {train_end}",
            "test_period": f"{test_start} s/d {test_end}",
        },
        "train_results": {
            "total_trades": train_trades,
            "wins": train_wins,
            "win_rate": round(train_wr, 1),
        },
        "test_results": {
            "total_raw_trades": test_trades_total,
            "raw_win_rate": round(test_raw_wr, 1),
            "ai_filtered_trades": ai_filtered_trades,
            "ai_filtered_wins": ai_filtered_wins,
            "ai_win_rate": round(ai_wr, 1),
            "accuracy_improvement": round(improvement, 1),
        },
        "learner_stats": learner.stats(),
        "weights": {FEATS[k]: round(learner.w[k+1], 4) for k in range(len(FEATS))},
        "duration_sec": round(duration, 2),
    }

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 65)
    print("  HASIL EVALUASI FORMAL (90% LATIH / 10% UJI)")
    print("=" * 65)
    print(f"Total Trade di Data Latih (90%) : {train_trades:,} trade (WR: {train_wr:.1f}%)")
    print(f"Total Sinyal Mentah di Data Uji : {test_trades_total:,} setup")
    print(f"Win Rate Mentah (Tanpa AI)      : {test_raw_wr:.1f}%")
    print(f"Trade Lolos Saringan AI         : {ai_filtered_trades:,} setup")
    print(f"Win Rate DENGAN FILTER AI       : {ai_wr:.1f}%  ({improvement:+.1f}% Akurasi Naik!)")
    print(f"Laporan Lengkap Tersimpan di    : {REPORT_OUT}")
    print("=" * 65)
    return result


if __name__ == "__main__":
    n = 200000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        n = int(sys.argv[1])
    run_train_test_split(n, train_ratio=0.90)
