"""Pelatihan Model AI Finex Bot menggunakan Dataset Historis XAUUSD (2004 - 2026).

Mengevaluasi ribuan setup EMA Ribbon (EMA 8 / 21 / 50) pada data pasar nyata,
mengekstrak fitur Welford-calibrated, dan memperbarui bobot Learner
sehingga bot langsung matang dan siap memfilter peluang dengan presisi tinggi.
"""
from __future__ import annotations

import math
import os
import sys
import time
import numpy as np
import pandas as pd

from learner import Learner, FEATS

DATA_PATH = r"d:\BigBoz\FinexTrade\finex-bot\data\historical_xauusd\XAU_5m_data.csv"
STATE_OUT = r"d:\BigBoz\FinexTrade\finex-bot\learner_state.json"
BACKUP_OUT = r"d:\BigBoz\FinexTrade\finex-bot\data\learner_state_xau_master.json"


def train_model(sample_limit: int = 150000) -> dict:
    print(f"=== MEMULAI PELATIHAN MODEL AI XAUUSD ===")
    print(f"Sumber Data: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File data tidak ditemukan: {DATA_PATH}")

    # Baca baris terbaru dari dataset (misal 150,000 bar terakhir untuk kecepatan & relevansi pasar modern)
    t0 = time.time()
    print(f"Membaca {sample_limit:,} lilin M5 terkini...")
    
    # Hitung total baris untuk mengambil tail
    with open(DATA_PATH, "rb") as fp:
        total_rows = sum(1 for _ in fp) - 1
    skip = max(1, total_rows - sample_limit)
    
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        skiprows=range(1, skip),
        names=["Date", "Open", "High", "Low", "Close", "Volume"],
        header=0,
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    df = df.sort_values("Date").reset_index(drop=True)
    t_read = time.time() - t0
    print(f"Berhasil memuat {len(df):,} bar (dari {df['Date'].iloc[0]} hingga {df['Date'].iloc[-1]}) dalam {t_read:.2f}s")

    # Hitung Indikator: EMA 8, EMA 21, EMA 50, ATR 14
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

    pip = 0.10  # Ukuran pip XAUUSD
    spread_pips = 1.8  # Spread realistis Finex / broker standar

    learner = Learner(STATE_OUT)
    trades_simulated = 0
    wins_count = 0
    losses_count = 0

    print("Mengekstraksi setup EMA Ribbon & melatih model online...")
    # Mulai evaluasi dari bar ke-60 agar indikator matang
    n_bars = len(df)
    last_trade_idx = 0

    for i in range(60, n_bars - 45):
        # Beri jeda minimal 3 bar antar trade agar tidak tumpang tindih
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
        body_i = c_i - o_i
        upper_wick = h_i - max(c_i, o_i)
        lower_wick = min(c_i, o_i) - l_i

        is_bull_align = (ema8_i > ema21_i and ema21_i > ema50_i)
        is_bear_align = (ema8_i < ema21_i and ema21_i < ema50_i)

        action = None
        side = 0
        entry = c_i
        sl = 0.0
        tp = 0.0
        wick_ratio = 0.0
        zone_depth = 0.5

        if is_bull_align:
            # Retest ke dynamic pocket EMA 8/21 atau kelanjutan momentum
            if l_i <= (ema8_i + 1.5 * pip) and c_i >= (ema21_i - 1.0 * pip):
                action = "BUY"
                side = 1
                sl = min(ema21_i - 2.5 * pip, l_i - 1.5 * pip)
                risk = max(entry - sl, 1.2 * atr_i)
                sl = entry - risk
                tp = entry + (1.6 * risk)  # Target RR 1:1.6
                wick_ratio = lower_wick / rng_i
                channel = max(ema8_i - ema50_i, 1e-4)
                zone_depth = np.clip((entry - ema50_i) / channel, 0.0, 1.5)
        elif is_bear_align:
            # Retest ke dynamic resistance EMA 8/21 atau kelanjutan momentum
            if h_i >= (ema8_i - 1.5 * pip) and c_i <= (ema21_i + 1.0 * pip):
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

        # Simulasikan hasil trade ke depan (horizon maksimal 40 bar = ~3.3 jam)
        win = False
        resolved = False
        risk_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)

        for fwd in range(1, 41):
            fwd_idx = i + fwd
            if fwd_idx >= n_bars:
                break
            f_h = float(h.iloc[fwd_idx])
            f_l = float(l.iloc[fwd_idx])

            if side == 1:  # BUY
                if f_h >= tp:
                    win = True
                    resolved = True
                    break
                elif f_l <= sl:
                    win = False
                    resolved = True
                    break
            else:  # SELL
                if f_l <= tp:
                    win = True
                    resolved = True
                    break
                elif f_h >= sl:
                    win = False
                    resolved = True
                    break

        if not resolved:
            # Jika waktu habis, evaluasi harga penutupan bar terakhir
            exit_c = float(c.iloc[min(i + 40, n_bars - 1)])
            win = (exit_c > entry) if side == 1 else (exit_c < entry)

        # Buat feature vector sesuai spesifikasi learner.py
        feat = {
            "trend": float(side),
            "zone_depth": float(zone_depth),
            "wick": float(wick_ratio),
            "rr": float(tp_dist / risk_dist) if risk_dist > 0 else 1.5,
            "sl_atr": float(risk_dist / atr_i),
            "hour_sin": math.sin(2 * math.pi * hr_i / 24),
            "hour_cos": math.cos(2 * math.pi * hr_i / 24),
            "spread": float(spread_pips),
        }

        learner.update(feat, win)
        trades_simulated += 1
        if win:
            wins_count += 1
        else:
            losses_count += 1
        last_trade_idx = i

    t_total = time.time() - t0
    win_rate = (wins_count / trades_simulated * 100.0) if trades_simulated > 0 else 0.0

    print(f"\n=== HASIL PELATIHAN AI MODEL ===")
    print(f"Total Trade Terlatih : {trades_simulated:,} trade")
    print(f"Kemenangan (Win)     : {wins_count:,} ({win_rate:.1f}%)")
    print(f"Kekalahan (Loss)     : {losses_count:,}")
    print(f"Total Sampel Akhir   : {learner.n:,}")
    print(f"Status Model Learner : {'AKTIF & MATANG' if learner.ready else 'BELUM MATANG'}")
    print(f"Waktu Eksekusi       : {t_total:.2f} detik")
    print(f"Bobot Fitur Model (w): {learner.w.tolist()}")

    # Simpan juga ke backup master
    try:
        import shutil
        shutil.copy(STATE_OUT, BACKUP_OUT)
        print(f"Backup master tersimpan di: {BACKUP_OUT}")
    except Exception as e:
        print(f"Gagal backup: {e}")

    return {
        "success": True,
        "trades": trades_simulated,
        "wins": wins_count,
        "win_rate": round(win_rate, 1),
        "total_samples": learner.n,
        "duration_sec": round(t_total, 2),
        "weights": learner.w.tolist(),
    }


if __name__ == "__main__":
    limit = 150000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        limit = int(sys.argv[1])
    train_model(limit)
