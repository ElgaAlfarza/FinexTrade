"""Modul Migrasi dan Sanitasi Jurnal Transaksi (P1-7 Data Integrity).

Tugas:
1. Menyelaraskan baris 11-kolom dan 12-kolom pada CSV riwayat lama.
2. Memperbaiki header trade_records_archive.csv agar menyertakan kolom 'technique'.
3. Memigrasikan seluruh riwayat unik ke tabel SQLite `trades` pada `data/bot_state.db`.
4. Menghasilkan checksum SHA-256 untuk verifikasi integritas data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("finex.migration")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CANONICAL_FIELDNAMES = [
    "ticket",
    "symbol",
    "side",
    "lots",
    "open_price",
    "close_price",
    "pnl",
    "source",
    "reason",
    "technique",
    "open_time",
    "close_time",
]


def infer_technique(reason: str, comment: str = "") -> str:
    txt = f"{comment} {reason}".upper()
    if "CPI" in txt or "PPI" in txt or "SNIPER" in txt:
        return "CPI_SNIPER"
    if "FIBO" in txt or "FIB" in txt:
        return "FIBO"
    if "EMA" in txt or "MA_TREND" in txt:
        return "EMA_TREND"
    if any(k in txt for k in ("PRICE_ACTION", "PRICE ACTION", "BREAKOUT", "PINBAR", "HAMMER", "IMPULSE")):
        return "PRICE_ACTION"
    if any(k in txt for k in ("RSI", "MOMENTUM", "OVERSOLD", "OVERBOUGHT", "REBOUND")):
        return "RSI_MOMENTUM"
    if "SMA" in txt or "KEYLEVEL" in txt or "KEY LEVEL" in txt:
        return "SMA_KEYLEVEL"
    return "EMA_TREND"


def parse_csv_row_robust(row: List[str], header: List[str]) -> Optional[Dict[str, Any]]:
    """Parse satu baris CSV riwayat secara tangguh terlepas dari variasi 11 vs 12 kolom."""
    if not row or len(row) < 9:
        return None

    # Jika baris adalah header itu sendiri, lewati
    if row[0].strip().lower() == "ticket":
        return None

    try:
        ticket = int(row[0].strip())
    except Exception:
        return None

    symbol = row[1].strip() if len(row) > 1 else "XAUUSD"
    side = row[2].strip().upper() if len(row) > 2 else "BUY"

    def _to_float(v: str, default: float = 0.0) -> float:
        try:
            return float(v.strip())
        except Exception:
            return default

    lots = _to_float(row[3]) if len(row) > 3 else 0.01
    open_price = _to_float(row[4]) if len(row) > 4 else 0.0
    close_price = _to_float(row[5]) if len(row) > 5 else 0.0
    pnl = _to_float(row[6]) if len(row) > 6 else 0.0
    source = row[7].strip() if len(row) > 7 else "BOT"
    reason = row[8].strip() if len(row) > 8 else ""

    technique = "EMA_TREND"
    open_time = ""
    close_time = ""

    if len(row) >= 12:
        # Format 12-kolom: ticket, symbol, side, lots, open_price, close_price, pnl, source, reason, technique, open_time, close_time
        technique = row[9].strip() or infer_technique(reason)
        open_time = row[10].strip()
        close_time = row[11].strip()
    elif len(row) == 11:
        # Format 11-kolom lama: tanpa 'technique' di indeks 9
        # Periksa apakah row[9] adalah tanggal (berisi '-' dan ':')
        if "-" in row[9] and ":" in row[9]:
            open_time = row[9].strip()
            close_time = row[10].strip()
            technique = infer_technique(reason)
        else:
            technique = row[9].strip() or infer_technique(reason)
            close_time = row[10].strip()
    elif len(row) == 10:
        technique = infer_technique(reason)
        close_time = row[9].strip()

    return {
        "ticket": ticket,
        "symbol": symbol,
        "side": side,
        "lots": lots,
        "open_price": open_price,
        "close_price": close_price,
        "pnl": pnl,
        "source": source,
        "reason": reason,
        "technique": technique,
        "open_time": open_time,
        "close_time": close_time,
    }


def sanitize_archive_csv(csv_path: str) -> int:
    """Perbaiki file CSV archive agar memiliki header canonical 12-kolom dan data bersih."""
    if not os.path.exists(csv_path):
        return 0

    rows_cleaned = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0

        for r in reader:
            parsed = parse_csv_row_robust(r, header)
            if parsed:
                rows_cleaned.append(parsed)

    # Tulis ulang file dengan header lengkap 12 kolom
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDNAMES)
        writer.writeheader()
        for item in rows_cleaned:
            writer.writerow(item)

    logger.info(f"Sanitasi {csv_path} selesai: {len(rows_cleaned)} transaksi dinormalisasi ke 12 kolom.")
    return len(rows_cleaned)


def init_trades_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            ticket INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            lots REAL NOT NULL,
            open_price REAL NOT NULL,
            close_price REAL NOT NULL,
            sl REAL,
            tp REAL,
            pnl REAL NOT NULL,
            r REAL,
            source TEXT,
            reason TEXT,
            technique TEXT,
            open_time TEXT,
            close_time TEXT,
            commission REAL DEFAULT 0.0,
            swap REAL DEFAULT 0.0,
            created_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time)")
    conn.commit()


def migrate_all_to_sqlite(data_dir: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Migrasikan seluruh arsip dan live trade ke database SQLite terstruktur."""
    db_file = db_path or os.path.join(data_dir, "bot_state.db")
    os.makedirs(os.path.dirname(db_file), exist_ok=True)

    archive_csv = os.path.join(data_dir, "trade_records_archive.csv")
    live_csv = os.path.join(data_dir, "trade_records.csv")
    archive_jsonl = os.path.join(data_dir, "trade_records_archive.jsonl")
    live_jsonl = os.path.join(data_dir, "trade_records.jsonl")

    # 1. Sanitasi CSV archive terlebih dahulu
    fixed_archive_count = sanitize_archive_csv(archive_csv) if os.path.exists(archive_csv) else 0

    all_trades: Dict[int, Dict[str, Any]] = {}

    def _ingest_csv(path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                hdr = next(reader)
            except StopIteration:
                return
            for r in reader:
                p = parse_csv_row_robust(r, hdr)
                if p:
                    all_trades[p["ticket"]] = p

    def _ingest_jsonl(path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    d = json.loads(line_str)
                    tk = int(d.get("ticket", 0))
                    if tk <= 0:
                        continue
                    entry = {
                        "ticket": tk,
                        "symbol": str(d.get("symbol", "XAUUSD")),
                        "side": str(d.get("side", "BUY")),
                        "lots": float(d.get("lots", 0.01) or 0.01),
                        "open_price": float(d.get("open_price", 0.0) or 0.0),
                        "close_price": float(d.get("close_price", 0.0) or 0.0),
                        "sl": float(d.get("sl")) if d.get("sl") is not None and str(d.get("sl")).strip() != "" else None,
                        "tp": float(d.get("tp")) if d.get("tp") is not None and str(d.get("tp")) != "" else None,
                        "pnl": float(d.get("pnl", 0.0) or 0.0),
                        "r": float(d.get("r")) if d.get("r") is not None and str(d.get("r")).strip() != "" else None,
                        "source": str(d.get("source", "BOT")),
                        "reason": str(d.get("reason", "")),
                        "technique": str(d.get("technique") or infer_technique(d.get("reason", ""), d.get("comment", ""))),
                        "open_time": str(d.get("open_time", "")),
                        "close_time": str(d.get("close_time", "")),
                        "commission": float(d.get("commission", 0.0) or 0.0),
                        "swap": float(d.get("swap", 0.0) or 0.0),
                    }
                    if tk in all_trades:
                        all_trades[tk].update({k: v for k, v in entry.items() if v is not None and v != ""})
                    else:
                        all_trades[tk] = entry
                except Exception:
                    continue

    # Ingest urut: archive CSV -> archive JSONL -> live CSV -> live JSONL
    _ingest_csv(archive_csv)
    _ingest_jsonl(archive_jsonl)
    _ingest_csv(live_csv)
    _ingest_jsonl(live_jsonl)

    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_file) as conn:
        init_trades_table(conn)
        for t in all_trades.values():
            conn.execute("""
                INSERT OR REPLACE INTO trades (
                    ticket, symbol, side, lots, open_price, close_price,
                    sl, tp, pnl, r, source, reason, technique,
                    open_time, close_time, commission, swap, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t["ticket"],
                t.get("symbol", "XAUUSD"),
                t.get("side", "BUY"),
                t.get("lots", 0.01),
                t.get("open_price", 0.0),
                t.get("close_price", 0.0),
                t.get("sl"),
                t.get("tp"),
                t.get("pnl", 0.0),
                t.get("r"),
                t.get("source", "BOT"),
                t.get("reason", ""),
                t.get("technique", "EMA_TREND"),
                t.get("open_time", ""),
                t.get("close_time", ""),
                t.get("commission", 0.0),
                t.get("swap", 0.0),
                now_iso,
            ))
        conn.commit()

        cur = conn.execute("SELECT COUNT(*) FROM trades")
        total_in_db = cur.fetchone()[0]

    # Hitung checksum data
    hasher = hashlib.sha256()
    for tk in sorted(all_trades.keys()):
        item = all_trades[tk]
        hasher.update(f"{tk}:{item.get('pnl')}:{item.get('close_time')}".encode("utf-8"))
    sha = hasher.hexdigest()

    summary = {
        "status": "SUCCESS",
        "archive_fixed": fixed_archive_count,
        "unique_trades_migrated": len(all_trades),
        "sqlite_total_rows": total_in_db,
        "sha256": sha,
        "db_path": db_file,
    }
    logger.info(f"Migrasi database SQLite selesai: {summary}")
    return summary


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    res = migrate_all_to_sqlite(base_dir)
    print("\nHASIL MIGRASI:")
    for k, v in res.items():
        print(f"  {k}: {v}")
