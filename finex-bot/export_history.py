"""Tahap 1: cek koneksi MT5 demo dan ekspor data candle ke CSV.

Jalankan di PC Windows yang terpasang MT5 dan sudah login ke akun demo Finex.

    python export_history.py --check
    python export_history.py --symbol EURUSD --tf M5 --from 2025-01-01 --to 2026-11-01 --out data/EURUSD_M5.csv

Kredensial dibaca dari file .env (lihat .env.example). Jangan kirim file .env ke siapa pun.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from mt5_client import TIMEFRAMES, MT5Session, load_env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true", help="hanya cek koneksi dan status akun")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--tf", choices=list(TIMEFRAMES), default="M5")
    ap.add_argument("--from", dest="date_from", default="2025-01-01")
    ap.add_argument("--to", dest="date_to", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out")
    a = ap.parse_args()

    load_env()
    try:
        login = int(os.environ["MT5_LOGIN"])
        password, server = os.environ["MT5_PASSWORD"], os.environ["MT5_SERVER"]
    except (KeyError, ValueError):
        raise SystemExit("Isi MT5_LOGIN, MT5_PASSWORD, MT5_SERVER di file .env (lihat .env.example)")

    with MT5Session(login, password, server, os.environ.get("MT5_PATH") or None) as s:
        for k, v in s.account_summary().items():
            print(f"{k:28}: {v}")
        if a.check:
            return
        d_from = datetime.strptime(a.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        d_to = datetime.strptime(a.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        out = a.out or f"data/{a.symbol}_{a.tf}.csv"
        n = s.export_rates(a.symbol, a.tf, d_from, d_to, out)
        print(f"\n{n} candle {a.symbol} {a.tf} disimpan ke {out}")
        print(f"Nilai per pip per 1 lot: {s.pip_value_per_lot(a.symbol, 0.0001):.4f} (untuk pip 0.0001)")


if __name__ == "__main__":
    main()
