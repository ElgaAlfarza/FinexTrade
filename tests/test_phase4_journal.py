"""Unit tests for Phase 4: Integritas Data dan Jurnal (P1-7, P2-1).

Verifies:
1. Thread-safety of PaperExecutor and unique non-colliding tickets under concurrency.
2. Robust parsing of 11-column vs 12-column rows without column shifting.
3. Archive CSV sanitization to canonical 12-column header.
4. Structured SQLite `trades` table persistence and deduplication.
"""
from __future__ import annotations

import csv
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest

BOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "finex-bot"))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

from tests.mock_mt5 import MockMT5
sys.modules["MetaTrader5"] = MockMT5()

from execution import PaperExecutor, save_trade_to_db
from migrate_trades import (
    CANONICAL_FIELDNAMES,
    parse_csv_row_robust,
    sanitize_archive_csv,
    migrate_all_to_sqlite,
)


def test_paper_executor_concurrent_unique_tickets():
    """Verify PaperExecutor is thread-safe and never generates duplicate tickets."""
    executor = PaperExecutor(initial_balance=100000.0)
    executor.update_prices("XAUUSD", 2650.0, 2650.25)

    num_threads = 10
    orders_per_thread = 50
    total_expected = num_threads * orders_per_thread

    created_positions = []
    lock = threading.Lock()

    def _worker(worker_id: int):
        for i in range(orders_per_thread):
            is_buy = (i % 2 == 0)
            sl_price = 2500.0 if is_buy else 2800.0
            tp_price = 2800.0 if is_buy else 2500.0
            pos = executor.open_trade(
                symbol="XAUUSD",
                side="BUY" if is_buy else "SELL",
                lots=0.01,
                sl=sl_price,
                tp=tp_price,
                risk_pips=100.0,
                comment=f"w{worker_id}_o{i}",
            )
            if pos:
                with lock:
                    created_positions.append(pos)
                # Concurrent price update within safe range & sl/tp modification
                if i % 5 == 0:
                    executor.update_prices("XAUUSD", 2650.5, 2650.75)
                    executor.modify_sl_tp(pos.ticket, sl_price - 1.0 if is_buy else sl_price + 1.0, tp_price)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(created_positions) == total_expected
    tickets = [p.ticket for p in created_positions]
    unique_tickets = set(tickets)
    assert len(unique_tickets) == total_expected, f"Duplicate tickets found! {total_expected - len(unique_tickets)} collisions"

    # Close all positions concurrently
    closed_count = executor.close_all(reason="TEST_CLOSE")
    assert closed_count == total_expected
    assert len(executor.get_positions()) == 0


def test_parse_csv_row_robust_11_vs_12_columns():
    """Verify 11-column legacy rows and 12-column rows are parsed correctly without column shifts."""
    # 1. Row with 12 columns (standard format with technique at index 9)
    row_12 = [
        "1001", "XAUUSD", "BUY", "0.02", "2650.0", "2655.0", "10.0",
        "BOT", "TP_HIT", "EMA_TREND", "2026-11-20 10:00:00", "2026-11-20 10:05:00"
    ]
    p12 = parse_csv_row_robust(row_12, CANONICAL_FIELDNAMES)
    assert p12 is not None
    assert p12["ticket"] == 1001
    assert p12["symbol"] == "XAUUSD"
    assert p12["technique"] == "EMA_TREND"
    assert p12["open_time"] == "2026-11-20 10:00:00"
    assert p12["close_time"] == "2026-11-20 10:05:00"

    # 2. Row with 11 columns (legacy format missing technique, row[9] is open_time)
    row_11 = [
        "1002", "EURUSD", "SELL", "0.05", "1.0850", "1.0830", "10.0",
        "MANUAL", "MANUAL_CLOSE", "2026-11-19 14:00:00", "2026-11-19 14:10:00"
    ]
    p11 = parse_csv_row_robust(row_11, ["ticket", "symbol", "side", "lots", "open_price", "close_price", "pnl", "source", "reason", "open_time", "close_time"])
    assert p11 is not None
    assert p11["ticket"] == 1002
    assert p11["open_time"] == "2026-11-19 14:00:00"
    assert p11["close_time"] == "2026-11-19 14:10:00"
    assert p11["technique"] == "EMA_TREND"  # Inferred, not shifted into dates


def test_sanitize_archive_csv(tmp_path):
    """Verify sanitize_archive_csv converts 11-col header and rows into clean 12-col canonical CSV."""
    test_csv = str(tmp_path / "test_archive.csv")
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        # Header has 11 columns
        f.write("ticket,symbol,side,lots,open_price,close_price,pnl,source,reason,open_time,close_time\n")
        # Row 1 has 12 columns
        f.write("2001,XAUUSD,BUY,0.01,2650.0,2655.0,5.0,BOT,TP_HIT,EMA_TREND,2026-11-20 08:00:00,2026-11-20 08:05:00\n")
        # Row 2 has 11 columns
        f.write("2002,XAUUSD,SELL,0.01,2655.0,2650.0,5.0,MANUAL,Cls PROFIT,2026-11-20 08:10:00,2026-11-20 08:15:00\n")

    cleaned_count = sanitize_archive_csv(test_csv)
    assert cleaned_count == 2

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == CANONICAL_FIELDNAMES
        r1 = next(reader)
        assert len(r1) == 12
        assert r1[9] == "EMA_TREND"
        assert r1[10] == "2026-11-20 08:00:00"
        r2 = next(reader)
        assert len(r2) == 12
        assert r2[10] == "2026-11-20 08:10:00"


def test_sqlite_trades_table_persistence(tmp_path):
    """Verify save_trade_to_db inserts, updates, and deduplicates in SQLite table `trades`."""
    db_file = str(tmp_path / "test_trades.db")
    trade = {
        "ticket": 3001,
        "symbol": "XAUUSD",
        "side": "BUY",
        "lots": 0.02,
        "open_price": 2650.0,
        "close_price": 2655.0,
        "sl": 2640.0,
        "tp": 2660.0,
        "pnl": 10.0,
        "r": 1.0,
        "source": "BOT",
        "reason": "TP_HIT",
        "technique": "EMA_TREND",
        "open_time": "2026-11-20 09:00:00",
        "close_time": "2026-11-20 09:10:00",
    }
    save_trade_to_db(trade, db_path=db_file)

    with sqlite3.connect(db_file) as conn:
        cur = conn.execute("SELECT ticket, symbol, side, lots, pnl, technique FROM trades WHERE ticket = 3001")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 3001
        assert row[1] == "XAUUSD"
        assert row[2] == "BUY"
        assert row[3] == 0.02
        assert row[4] == 10.0
        assert row[5] == "EMA_TREND"

    # Update same trade (e.g. commission added)
    trade["commission"] = -0.5
    save_trade_to_db(trade, db_path=db_file)

    with sqlite3.connect(db_file) as conn:
        cur = conn.execute("SELECT COUNT(*), commission FROM trades WHERE ticket = 3001")
        count, comm = cur.fetchone()
        assert count == 1
        assert comm == -0.5
