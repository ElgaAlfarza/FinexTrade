"""Tahap 3: Engine Eksekusi Order (Execution & Position Manager).

Mendukung dua backend:
1. `MT5Executor`: Eksekusi langsung ke terminal MetaTrader 5 Desktop (IPC).
2. `PaperExecutor`: Simulator paper-trading lokal berpresisi tinggi.

Fitur Trader:
- Auto lot sizing dari risk.py
- Trailing Stop & Auto Break-Even (kunci modal saat +1R)
- Proteksi spread (tolak eksekusi jika spread meledak)
- Circuit Breaker harian (max daily loss)
- Emergency Kill Switch (tutup semua posisi seketika)
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import SYMBOL_SPECS, StrategyParams, symbol_decimals
from risk import can_trade_today, estimate_pip_value, lot_size

logger = logging.getLogger("finex.execution")

TRADE_RECORDS_CSV = os.path.join(os.path.dirname(__file__), "data", "trade_records.csv")
TRADE_RECORDS_JSONL = os.path.join(os.path.dirname(__file__), "data", "trade_records.jsonl")
TRADE_RECORDS_ARCHIVE_CSV = os.path.join(os.path.dirname(__file__), "data", "trade_records_archive.csv")
TRADE_RECORDS_ARCHIVE_JSONL = os.path.join(os.path.dirname(__file__), "data", "trade_records_archive.jsonl")
SESSION_RESET_FILE = os.path.join(os.path.dirname(__file__), "data", "session_reset.json")
_trade_records_lock = threading.Lock()


def get_session_reset_info() -> dict:
    if os.path.exists(SESSION_RESET_FILE):
        try:
            with open(SESSION_RESET_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def set_session_reset(reset_time: Optional[float] = None, initial_balance: Optional[float] = None, mt5_instance: Any = None) -> dict:
    if reset_time is None:
        reset_time = time.time()
        if mt5_instance:
            try:
                tick = mt5_instance.symbol_info_tick("EURUSD")
                if tick and getattr(tick, "time", 0) > 0:
                    reset_time = max(reset_time, float(tick.time))
                to_date = datetime.now() + timedelta(days=2)
                from_date = datetime.now() - timedelta(days=1)
                deals = mt5_instance.history_deals_get(from_date, to_date)
                if deals:
                    max_d = max(getattr(d, "time", 0) for d in deals)
                    if max_d > 0:
                        reset_time = max(reset_time, float(max_d) + 1.0)
            except Exception:
                pass
    dt_str = datetime.fromtimestamp(reset_time).strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "reset_time": float(reset_time),
        "reset_time_str": dt_str,
        "initial_balance": float(initial_balance) if initial_balance is not None else None,
    }
    os.makedirs(os.path.dirname(SESSION_RESET_FILE), exist_ok=True)
    with open(SESSION_RESET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Arsipkan riwayat transaksi lama ke archive agar data latih bot 100% terjaga utuh
    with _trade_records_lock:
        try:
            import shutil
            if os.path.exists(TRADE_RECORDS_CSV):
                if not os.path.exists(TRADE_RECORDS_ARCHIVE_CSV) or os.path.getsize(TRADE_RECORDS_ARCHIVE_CSV) < os.path.getsize(TRADE_RECORDS_CSV):
                    shutil.copy2(TRADE_RECORDS_CSV, TRADE_RECORDS_ARCHIVE_CSV)
            if os.path.exists(TRADE_RECORDS_JSONL):
                if not os.path.exists(TRADE_RECORDS_ARCHIVE_JSONL) or os.path.getsize(TRADE_RECORDS_ARCHIVE_JSONL) < os.path.getsize(TRADE_RECORDS_JSONL):
                    shutil.copy2(TRADE_RECORDS_JSONL, TRADE_RECORDS_ARCHIVE_JSONL)

            # Reset file transaksi sesi saat ini (kosongkan dengan header)
            fieldnames = ["ticket", "symbol", "side", "lots", "open_price", "close_price", "pnl", "source", "reason", "open_time", "close_time"]
            with open(TRADE_RECORDS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            with open(TRADE_RECORDS_JSONL, "w", encoding="utf-8") as f_json:
                pass
        except Exception as e:
            logger.warning(f"[SESSION_RESET] Gagal mengarsipkan riwayat: {e}")

    logger.info(f"[SESSION_RESET] Sesi di-reset ke {dt_str}. Saldo awal: ${initial_balance}. Data latih diarsipkan.")
    return data


def clear_session_reset() -> None:
    if os.path.exists(SESSION_RESET_FILE):
        try:
            os.remove(SESSION_RESET_FILE)
        except Exception:
            pass


def infer_technique_from_text(text: str) -> str:
    """Mengidentifikasi teknik yang melandasi transaksi dari komentar atau alasan eksekusi."""
    txt = str(text or "").upper()
    if "CPI" in txt or "PPI" in txt or "SNIPER" in txt:
        return "CPI_SNIPER"
    if "FIBO" in txt or "FIB" in txt:
        return "FIBO"
    if "EMA" in txt or "MA_TREND" in txt:
        return "EMA_TREND"
    if "PRICE_ACTION" in txt or "PRICE ACTION" in txt or "BREAKOUT" in txt or "PINBAR" in txt or "HAMMER" in txt or "IMPULSE" in txt:
        return "PRICE_ACTION"
    if "RSI" in txt or "MOMENTUM" in txt or "OVERSOLD" in txt or "OVERBOUGHT" in txt or "REBOUND" in txt:
        return "RSI_MOMENTUM"
    if "FORECAST" in txt or "WALK" in txt or "EWMA" in txt:
        return "AI_FORECAST"
    if "GEMINI" in txt or "AI GATE" in txt:
        return "AI_GEMINI"
    if "SMA_KEYLEVEL" in txt or "KEYLEVEL" in txt or "KEY LEVEL" in txt or "SMA 50" in txt or "SMA 200" in txt or "ENGULFING" in txt:
        return "SMA_KEYLEVEL"
    return "ALL"


def extract_techniques_from_trade(trade: dict) -> List[str]:
    """Mengidentifikasi teknik yang melandasi transaksi (Fokus 100% pada EMA_TREND)."""
    return ["EMA_TREND"]


def save_trade_record(trade: dict) -> None:
    """Mencatat transaksi ke file CSV & JSONL permanen (baik yang dieksekusi bot maupun manual)."""
    if not trade:
        return
    with _trade_records_lock:
        try:
            os.makedirs(os.path.dirname(TRADE_RECORDS_CSV), exist_ok=True)
            file_exists = os.path.exists(TRADE_RECORDS_CSV)
            existing_tickets = set()
            if file_exists:
                try:
                    with open(TRADE_RECORDS_CSV, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            t = r.get("ticket")
                            if t:
                                existing_tickets.add(str(t))
                except Exception:
                    pass

            ticket_str = str(trade.get("ticket", ""))
            if ticket_str and ticket_str not in existing_tickets:
                fieldnames = ["ticket", "symbol", "side", "lots", "open_price", "close_price", "pnl", "source", "reason", "technique", "open_time", "close_time"]
                with open(TRADE_RECORDS_CSV, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists or os.path.getsize(TRADE_RECORDS_CSV) == 0:
                        writer.writeheader()
                    row = {k: trade.get(k, "") for k in fieldnames}
                    if not row.get("technique"):
                        row["technique"] = infer_technique_from_text(f"{trade.get('comment', '')} {trade.get('reason', '')}")
                    writer.writerow(row)
                with open(TRADE_RECORDS_JSONL, "a", encoding="utf-8") as f_json:
                    f_json.write(json.dumps(trade, ensure_ascii=False) + "\n")
                logger.info(f"[RECORDER] Trade #{ticket_str} ({trade.get('source', 'TRADE')} / {row.get('technique')}) dicatat ke {TRADE_RECORDS_CSV}")
        except Exception as e:
            logger.warning(f"[RECORDER] Gagal mencatat trade ke file: {e}")


def sync_trade_records_to_file(trades: List[dict]) -> None:
    """Sinkronisasi batch seluruh riwayat trade (MT5 / Paper) ke file CSV & JSONL tanpa duplikasi."""
    if not trades:
        return
    with _trade_records_lock:
        try:
            os.makedirs(os.path.dirname(TRADE_RECORDS_CSV), exist_ok=True)
            file_exists = os.path.exists(TRADE_RECORDS_CSV)
            existing_tickets = set()
            if file_exists:
                try:
                    with open(TRADE_RECORDS_CSV, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            t = r.get("ticket")
                            if t:
                                existing_tickets.add(str(t))
                except Exception:
                    pass

            fieldnames = ["ticket", "symbol", "side", "lots", "open_price", "close_price", "pnl", "source", "reason", "technique", "open_time", "close_time"]
            new_rows = []
            new_json_lines = []
            for t in trades:
                ticket_str = str(t.get("ticket", ""))
                if ticket_str and ticket_str not in existing_tickets:
                    existing_tickets.add(ticket_str)
                    row = {k: t.get(k, "") for k in fieldnames}
                    if not row.get("technique"):
                        row["technique"] = infer_technique_from_text(f"{t.get('comment', '')} {t.get('reason', '')}")
                    new_rows.append(row)
                    new_json_lines.append(json.dumps(t, ensure_ascii=False))

            if new_rows:
                with open(TRADE_RECORDS_CSV, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists or os.path.getsize(TRADE_RECORDS_CSV) == 0:
                        writer.writeheader()
                    writer.writerows(new_rows)
                with open(TRADE_RECORDS_JSONL, "a", encoding="utf-8") as f_json:
                    for line in new_json_lines:
                        f_json.write(line + "\n")
                logger.info(f"[RECORDER] {len(new_rows)} riwayat transaksi (Bot & Manual) tersimpan permanen di {TRADE_RECORDS_CSV}")
        except Exception as e:
            logger.warning(f"[RECORDER] Gagal sinkronisasi catatan trade ke file: {e}")


def get_symbol_pip_size(symbol: str) -> float:
    """Ambil pip size spesifik per simbol aset."""
    spec = SYMBOL_SPECS.get(symbol.upper(), {})
    if "pip_size" in spec:
        return spec["pip_size"]
    sym = symbol.upper()
    if any(k in sym for k in ("BTC", "US100", "US500", "US30", "JP225")):
        return 1.0
    if "ETH" in sym or "XAU" in sym:
        return 0.1
    if "XAG" in sym or "JPY" in sym or "XTI" in sym:
        return 0.01
    return 0.0001


@dataclass
class Position:
    ticket: int
    symbol: str
    side: str            # "BUY" atau "SELL"
    lots: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    risk_pips: float
    r_target: float
    open_time: str
    be_active: bool = False
    floating_pnl: float = 0.0
    floating_r: float = 0.0
    peak_pnl: float = 0.0
    trailing_runner_active: bool = False
    comment: str = ""
    magic: int = 20261106
    technique: str = ""
    is_manual: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class IExecutor:
    """Interface umum eksekutor trading."""
    def get_account_info(self) -> dict:
        raise NotImplementedError

    def get_positions(self) -> List[Position]:
        raise NotImplementedError

    def open_trade(self, symbol: str, side: str, lots: float, sl: float, tp: float, risk_pips: float, comment: str = "") -> Optional[Position]:
        raise NotImplementedError

    def modify_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        raise NotImplementedError

    def close_trade(self, ticket: int, reason: str = "MANUAL") -> bool:
        raise NotImplementedError

    def close_all(self, reason: str = "KILL_SWITCH") -> int:
        raise NotImplementedError

    def update_prices(self, symbol: str, bid: float, ask: float) -> None:
        pass

    def get_trade_history(self) -> List[dict]:
        return []

    def get_symbol_trade_specs(self, symbol: str, pip_size: float, ref_price: float = 0.0) -> dict:
        return {
            "pip_value": estimate_pip_value(symbol, pip_size),
            "min_lot": 0.01,
            "lot_step": 0.01,
            "max_lot": 100.0,
            "digits": symbol_decimals(symbol),
        }

    def get_required_margin(self, symbol: str, side: str, lots: float, price: float = 0.0) -> float:
        contract = 100.0 if "XAU" in symbol or "GOLD" in symbol else 100000.0
        px = price if price > 0 else (4000.0 if "XAU" in symbol else 1.1)
        return round((float(lots) * contract * px) / 500.0, 2)


class PaperExecutor(IExecutor):
    """Simulator Paper Trading independen."""

    def get_required_margin(self, symbol: str, side: str, lots: float, price: float = 0.0) -> float:
        contract = 100.0 if "XAU" in symbol or "GOLD" in symbol else 100000.0
        px = price if price > 0 else (4000.0 if "XAU" in symbol else 1.1)
        return round((float(lots) * contract * px) / 500.0, 2)

    def __init__(self, initial_balance: float = 10000.0, currency: str = "USD"):
        self.balance: float = initial_balance
        self.equity: float = initial_balance
        self.currency: str = currency
        self.positions: Dict[int, Position] = {}
        self.trade_history: List[dict] = []
        self._next_ticket: int = 100001
        self.latest_prices: Dict[str, dict] = {} # {symbol: {bid, ask, time}}
        self.daily_start_balance: float = initial_balance
        self.monthly_start_balance: float = initial_balance
        self.last_day = datetime.now(timezone.utc).date()
        self.last_month = datetime.now(timezone.utc).month

    def get_account_info(self) -> dict:
        floating = sum(p.floating_pnl for p in self.positions.values())
        self.equity = round(self.balance + floating, 2)
        margin_used = sum(p.lots * 1000.0 for p in self.positions.values()) # est margin 1:100

        now_dt = datetime.now(timezone.utc)
        today_str = now_dt.strftime("%Y-%m-%d")
        month_str = now_dt.strftime("%Y-%m")
        if now_dt.date() != self.last_day or self.daily_start_balance <= 0:
            self.daily_start_balance = self.balance
            self.last_day = now_dt.date()
        if now_dt.month != self.last_month or self.monthly_start_balance <= 0:
            self.monthly_start_balance = self.balance
            self.last_month = now_dt.month

        reset_info = get_session_reset_info()
        reset_epoch = float(reset_info.get("reset_time", 0))

        realized_today = sum(t.get("pnl", 0) for t in self.trade_history if t.get("close_time", "").startswith(today_str) and (reset_epoch <= 0 or t.get("close_epoch", 0) >= reset_epoch))
        realized_month = sum(t.get("pnl", 0) for t in self.trade_history if t.get("close_time", "").startswith(month_str) and (reset_epoch <= 0 or t.get("close_epoch", 0) >= reset_epoch))
        profit_today = round(realized_today + floating, 2)
        profit_today_pct = round((profit_today / self.daily_start_balance) * 100, 2) if self.daily_start_balance > 0 else 0.0
        profit_month = round(realized_month + floating, 2)
        profit_month_pct = round((profit_month / self.monthly_start_balance) * 100, 2) if self.monthly_start_balance > 0 else 0.0
        est_balance_tomorrow = round(self.balance + floating, 2)

        return {
            "mode": "PAPER",
            "connected": True,
            "login": 888888,
            "server": "Local-PaperSimulator",
            "currency": self.currency,
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "margin": round(margin_used, 2),
            "free_margin": round(self.equity - margin_used, 2),
            "floating_pnl": round(floating, 2),
            "daily_start_balance": round(self.daily_start_balance, 2),
            "profit_today": profit_today,
            "profit_today_pct": profit_today_pct,
            "realized_today": round(realized_today, 2),
            "monthly_start_balance": round(self.monthly_start_balance, 2),
            "profit_month": profit_month,
            "profit_month_pct": profit_month_pct,
            "est_balance_tomorrow": est_balance_tomorrow,
            "open_positions": len(self.positions),
            "closed_trades": len(self.trade_history),
        }

    def update_prices(self, symbol: str, bid: float, ask: float) -> None:
        self.latest_prices[symbol] = {"bid": bid, "ask": ask, "time": datetime.now(timezone.utc).isoformat()}
        # Update floating pnl posisi yang terbuka
        for p in list(self.positions.values()):
            if p.symbol != symbol:
                continue
            cur = bid if p.side == "BUY" else ask
            p.current_price = cur
            pnl_price = (cur - p.open_price) if p.side == "BUY" else (p.open_price - cur)
            pip_val = estimate_pip_value(symbol)
            pip_size = get_symbol_pip_size(symbol)
            pips_gain = pnl_price / pip_size
            p.floating_pnl = round(pips_gain * pip_val * p.lots, 2)
            p.floating_r = round(pips_gain / p.risk_pips, 2) if p.risk_pips > 0 else 0.0

            # Cek apakah menyentuh SL atau TP pada update harga
            if p.side == "BUY":
                if bid <= p.sl:
                    self._execute_exit(p.ticket, p.sl, "SL")
                elif bid >= p.tp:
                    self._execute_exit(p.ticket, p.tp, "TP")
            else:
                if ask >= p.sl:
                    self._execute_exit(p.ticket, p.sl, "SL")
                elif ask <= p.tp:
                    self._execute_exit(p.ticket, p.tp, "TP")

    def _execute_exit(self, ticket: int, exit_price: float, reason: str) -> None:
        if ticket not in self.positions:
            return
        p = self.positions.pop(ticket)
        pnl_price = (exit_price - p.open_price) if p.side == "BUY" else (p.open_price - exit_price)
        pip_size = get_symbol_pip_size(p.symbol)
        pips_gain = pnl_price / pip_size
        pip_val = estimate_pip_value(p.symbol)
        realized_pnl = round(pips_gain * pip_val * p.lots, 2)
        r = round(pips_gain / p.risk_pips, 2) if p.risk_pips > 0 else 0.0
        self.balance = round(self.balance + realized_pnl, 2)
        self.equity = round(self.balance + sum(pos.floating_pnl for pos in self.positions.values()), 2)

        now_dt = datetime.now()
        now_epoch = int(time.time())
        is_manual = str(getattr(p, 'comment', '') or '').startswith("Manual")
        source = "MANUAL" if is_manual else "BOT"
        trade_entry = {
            "ticket": p.ticket,
            "symbol": p.symbol,
            "side": p.side,
            "lots": p.lots,
            "open_time": p.open_time,
            "close_time": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "close_epoch": now_epoch,
            "open_price": p.open_price,
            "close_price": exit_price,
            "sl": p.sl,
            "tp": p.tp,
            "r": r,
            "pnl": realized_pnl,
            "source": source,
            "reason": reason,
        }
        self.trade_history.append(trade_entry)
        save_trade_record(trade_entry)
        logger.info(f"[PAPER] Trade #{ticket} Closed @ {exit_price} ({reason}) | PnL: ${realized_pnl} ({r:+.2f}R)")

    def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_trade_history(self) -> List[dict]:
        res = list(reversed(self.trade_history))
        for t in res:
            if "source" not in t:
                t["source"] = "MANUAL" if str(t.get("comment", "")).startswith("Manual") else "BOT"
            if "close_epoch" not in t:
                t["close_epoch"] = int(time.time())
        res.sort(key=lambda x: (int(x.get("close_epoch", 0)), int(x.get("ticket", 0))), reverse=True)
        sync_trade_records_to_file(res)
        return res

    def open_trade(self, symbol: str, side: str, lots: float, sl: float, tp: float, risk_pips: float, comment: str = "") -> Optional[Position]:
        ticket = self._next_ticket
        self._next_ticket += 1
        spec = SYMBOL_SPECS.get(symbol.upper(), {})
        def_p = spec.get("default_price", 1.0850)
        spread = spec.get("spread_pips", 1.0) * spec.get("pip_size", 0.0001)
        quote = self.latest_prices.get(symbol, {"bid": def_p, "ask": def_p + spread})
        fill_price = quote["ask"] if side == "BUY" else quote["bid"]
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        pos = Position(
            ticket=ticket, symbol=symbol, side=side, lots=lots,
            open_price=fill_price, current_price=fill_price, sl=sl, tp=tp,
            risk_pips=risk_pips, r_target=abs(tp - fill_price) / abs(fill_price - sl) if abs(fill_price - sl) > 0 else 1.5,
            open_time=now_str, comment=comment
        )
        self.positions[ticket] = pos
        logger.info(f"[PAPER] Trade #{ticket} Open {side} {lots} lot @ {fill_price} SL:{sl} TP:{tp}")
        return pos

    def modify_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        if ticket in self.positions:
            self.positions[ticket].sl = new_sl
            self.positions[ticket].tp = new_tp
            logger.info(f"[PAPER] Trade #{ticket} SL/TP modified -> SL:{new_sl} TP:{new_tp}")
            return True
        return False

    def close_trade(self, ticket: int, reason: str = "MANUAL") -> bool:
        if ticket in self.positions:
            p = self.positions[ticket]
            self._execute_exit(ticket, p.current_price, reason)
            return True
        return False

    def close_all(self, reason: str = "KILL_SWITCH") -> int:
        tickets = list(self.positions.keys())
        for t in tickets:
            self.close_trade(t, reason)
        return len(tickets)


class MT5Executor(IExecutor):
    """Eksekutor Live menggunakan IPC MetaTrader 5 (thread-safe + deal cache)."""

    def __init__(self, mt5_session):
        self.s = mt5_session
        self.mt5 = mt5_session.mt5
        self.magic = 20261106
        self.closed_trades: List[dict] = []
        self.last_order_error: str = ""
        self._lock = threading.RLock()
        self._deals_cache_ts: float = 0.0
        self._deals_summary_cache: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._hist_cache_ts: float = 0.0
        self._hist_cache: List[dict] = []

    def _invalidate_caches(self) -> None:
        self._deals_cache_ts = 0.0
        self._hist_cache_ts = 0.0

    def get_symbol_trade_specs(self, symbol: str, pip_size: float, ref_price: float = 0.0) -> dict:
        fallback = super().get_symbol_trade_specs(symbol, pip_size, ref_price)
        with self._lock:
            try:
                if not self.mt5.symbol_select(symbol, True):
                    return fallback
                info = self.mt5.symbol_info(symbol)
                if info is None:
                    return fallback
                digits = int(getattr(info, "digits", fallback["digits"]))
                min_lot = float(getattr(info, "volume_min", 0.01) or 0.01)
                lot_step = float(getattr(info, "volume_step", 0.01) or 0.01)
                max_lot = float(getattr(info, "volume_max", 100.0) or 100.0)

                px = float(ref_price)
                if px <= 0:
                    tick = self.mt5.symbol_info_tick(symbol)
                    px = float(tick.ask or tick.bid) if tick and (tick.ask or tick.bid) else float(getattr(info, "bid", 0.0) or 0.0)
                pip_val = fallback["pip_value"]
                if px > 0 and pip_size > 0:
                    calc = self.mt5.order_calc_profit(self.mt5.ORDER_TYPE_BUY, symbol, 1.0, px, px + float(pip_size))
                    if calc is not None and calc > 0:
                        pip_val = float(calc)
                return {
                    "pip_value": pip_val,
                    "min_lot": min_lot,
                    "lot_step": lot_step,
                    "max_lot": max_lot,
                    "digits": digits,
                }
            except Exception:
                return fallback

    def get_account_info(self) -> dict:
        with self._lock:
            info = self.mt5.account_info()
            t = self.mt5.terminal_info()
            if info is None:
                return {"connected": False, "mode": "MT5_DISCONNECTED"}

            now_mono = time.monotonic()
            if now_mono - self._deals_cache_ts >= 3.0:
                # Ambil waktu server MT5 dari tick terkini agar sinkron dengan broker Finex
                tick = None
                try:
                    sym = getattr(self, "symbol", "EURUSD") or "EURUSD"
                    tick = self.mt5.symbol_info_tick(sym)
                    if tick is None:
                        tick = self.mt5.symbol_info_tick("EURUSD")
                except Exception:
                    tick = None

                server_now = datetime.fromtimestamp(tick.time) if tick and getattr(tick, "time", 0) > 0 else datetime.now()

                # PENTING: to_date diberi buffer masa depan (+2 hari) agar transaksi yang
                # dicatat broker tidak terpotong akibat perbedaan zona waktu (GMT+2/GMT+3 vs jam lokal PC)
                to_date = server_now + timedelta(days=2)
                today_start = datetime(server_now.year, server_now.month, server_now.day, 0, 0, 0)
                month_start = datetime(server_now.year, server_now.month, 1, 0, 0, 0)
                days_30_start = server_now - timedelta(days=30)
                period_start = min(month_start, days_30_start)

                today_fetch_start = today_start
                period_fetch_start = period_start

                deals_today = self.mt5.history_deals_get(today_fetch_start, to_date) or []
                deals_month = self.mt5.history_deals_get(period_fetch_start, to_date) or []

                reset_info = get_session_reset_info()
                reset_epoch = float(reset_info.get("reset_time", 0))
                initial_balance = reset_info.get("initial_balance")

                if reset_epoch > 0:
                    deals_today = [d for d in deals_today if getattr(d, 'time', 0) >= reset_epoch]
                    deals_month = [d for d in deals_month if getattr(d, 'time', 0) >= reset_epoch]

                realized_today = sum(
                    d.profit + getattr(d, 'swap', 0) + getattr(d, 'commission', 0)
                    for d in deals_today
                    if getattr(d, 'entry', 0) == self.mt5.DEAL_ENTRY_OUT
                )
                deposits_today = sum(
                    d.profit for d in deals_today
                    if getattr(d, 'type', 0) == self.mt5.DEAL_TYPE_BALANCE
                )
                realized_month = sum(
                    d.profit + getattr(d, 'swap', 0) + getattr(d, 'commission', 0)
                    for d in deals_month
                    if getattr(d, 'entry', 0) == self.mt5.DEAL_ENTRY_OUT
                )
                deposits_month = sum(
                    d.profit for d in deals_month
                    if getattr(d, 'type', 0) == self.mt5.DEAL_TYPE_BALANCE
                )
                self._deals_summary_cache = (realized_today, deposits_today, realized_month, deposits_month)
                self._deals_cache_ts = now_mono
            else:
                realized_today, deposits_today, realized_month, deposits_month = self._deals_summary_cache

        reset_info = get_session_reset_info()
        reset_epoch = float(reset_info.get("reset_time", 0))
        initial_balance = reset_info.get("initial_balance")

        if reset_epoch > 0 and initial_balance is not None:
            daily_start_balance = round(float(initial_balance), 2)
            monthly_start_balance = round(float(initial_balance), 2)
        else:
            daily_start_balance = round(info.balance - realized_today - deposits_today, 2)
            if daily_start_balance <= 0:
                daily_start_balance = round(info.balance, 2)

            if deposits_month > 0:
                monthly_start_balance = round(deposits_month, 2)
            else:
                monthly_start_balance = round(info.balance - realized_month, 2)
            if monthly_start_balance <= 0:
                monthly_start_balance = round(info.balance, 2)

        profit_today = round(realized_today + info.profit, 2)
        profit_today_pct = round((profit_today / daily_start_balance) * 100, 2) if daily_start_balance > 0 else 0.0

        profit_month = round(realized_month + info.profit, 2)
        profit_month_pct = round((profit_month / monthly_start_balance) * 100, 2) if monthly_start_balance > 0 else 0.0

        est_balance_tomorrow = round(info.balance + info.profit, 2)

        return {
            "mode": "MT5_DEMO" if info.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO else "MT5_REAL",
            "connected": True,
            "login": info.login,
            "server": info.server,
            "currency": info.currency,
            "balance": round(info.balance, 2),
            "equity": round(info.equity, 2),
            "margin": round(info.margin, 2),
            "free_margin": round(info.margin_free, 2),
            "floating_pnl": round(info.profit, 2),
            "daily_start_balance": daily_start_balance,
            "profit_today": profit_today,
            "profit_today_pct": profit_today_pct,
            "realized_today": round(realized_today, 2),
            "monthly_start_balance": monthly_start_balance,
            "profit_month": profit_month,
            "profit_month_pct": profit_month_pct,
            "est_balance_tomorrow": est_balance_tomorrow,
            "terminal_algo_ok": t.trade_allowed if t else False,
            "account_algo_ok": info.trade_expert,
        }

    def get_positions(self) -> List[Position]:
        with self._lock:
            mt5_pos = self.mt5.positions_get()
        if mt5_pos is None or len(mt5_pos) == 0:
            return []
        res = []
        for p in mt5_pos:
            side = "BUY" if p.type == self.mt5.POSITION_TYPE_BUY else "SELL"
            pip_sz = get_symbol_pip_size(p.symbol)
            r_pips = abs(p.price_open - p.sl) / pip_sz if p.sl > 0 and pip_sz > 0 else 20.0
            cur_pips = ((p.price_current - p.price_open) if side == "BUY" else (p.price_open - p.price_current)) / pip_sz if pip_sz > 0 else 0.0
            res.append(
                Position(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    side=side,
                    lots=p.volume,
                    open_price=p.price_open,
                    current_price=p.price_current,
                    sl=p.sl,
                    tp=p.tp,
                    risk_pips=r_pips,
                    r_target=1.5,
                    open_time=datetime.fromtimestamp(p.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    floating_pnl=round(p.profit, 2),
                    floating_r=round(cur_pips / r_pips, 2) if r_pips > 0 else 0.0,
                    magic=p.magic,
                    comment=p.comment,
                    technique=infer_technique_from_text(p.comment),
                )
            )
        return res

    def get_required_margin(self, symbol: str, side: str, lots: float, price: float = 0.0) -> float:
        with self._lock:
            try:
                order_type = self.mt5.ORDER_TYPE_BUY if str(side).upper() == "BUY" else self.mt5.ORDER_TYPE_SELL
                px = float(price)
                if px <= 0:
                    tick = self.mt5.symbol_info_tick(symbol)
                    px = float(tick.ask if order_type == self.mt5.ORDER_TYPE_BUY else tick.bid) if tick else 0.0
                if px > 0:
                    m = self.mt5.order_calc_margin(order_type, symbol, float(lots), float(px))
                    if m is not None and m > 0:
                        return float(round(m, 2))
            except Exception as e:
                logger.debug(f"Gagal order_calc_margin {symbol}: {e}")
        lev = 500.0
        try:
            acc = self.mt5.account_info()
            if acc and getattr(acc, "leverage", 0) > 0:
                lev = float(acc.leverage)
        except Exception:
            pass
        contract = 100.0 if "XAU" in symbol or "GOLD" in symbol else 100000.0
        px = price if price > 0 else (4000.0 if "XAU" in symbol else 1.1)
        return round((float(lots) * contract * px) / lev, 2)

    def open_trade(self, symbol: str, side: str, lots: float, sl: float, tp: float, risk_pips: float, comment: str = "") -> Optional[Position]:
        with self._lock:
            if not self.mt5.symbol_select(symbol, True):
                self.last_order_error = f"Simbol {symbol} tidak tersedia di server MT5 ini."
                logger.error(self.last_order_error)
                return None
            info = self.mt5.symbol_info(symbol)
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                self.last_order_error = f"Gagal mengambil tick harga untuk {symbol}"
                logger.error(self.last_order_error)
                return None

            digits = int(getattr(info, "digits", symbol_decimals(symbol))) if info else symbol_decimals(symbol)
            step = float(getattr(info, "volume_step", 0.01) or 0.01) if info else 0.01
            vmin = float(getattr(info, "volume_min", 0.01) or 0.01) if info else 0.01
            vmax = float(getattr(info, "volume_max", 100.0) or 100.0) if info else 100.0

            # Validasi Stop Loss Wajib (Aturan Keras Algoritmik)
            if sl is None or float(sl) <= 0:
                self.last_order_error = "Order DITOLAK: Stop Loss wajib ditentukan (dilarang membuka posisi tanpa SL)."
                logger.error(f"[MT5] {self.last_order_error}")
                return None

            # Validasi Ukuran Lot
            if lots is None or float(lots) <= 0:
                self.last_order_error = "Order DITOLAK: Ukuran lot 0 atau negatif (risiko melebihi batas modal)."
                logger.error(f"[MT5] {self.last_order_error}")
                return None
            if float(lots) < vmin:
                self.last_order_error = f"Order DITOLAK: Ukuran lot terhitung ({lots}) di bawah lot minimum broker ({vmin})."
                logger.warning(f"[MT5] {self.last_order_error}")
                return None

            norm_lots = round(min(max(math.floor(float(lots) / step + 1e-9) * step, vmin), vmax), 4)
            norm_sl = round(float(sl), digits)
            norm_tp = round(float(tp), digits) if tp and float(tp) > 0 else 0.0

            # Validasi stops_level / freeze_level broker
            stops_level = getattr(info, "trade_stops_level", 0) or 0
            freeze_level = getattr(info, "trade_freeze_level", 0) or 0
            point = getattr(info, "point", 0.01) or 0.01
            min_dist = max(stops_level, freeze_level) * point

            price = tick.ask if side == "BUY" else tick.bid
            order_type = self.mt5.ORDER_TYPE_BUY if side == "BUY" else self.mt5.ORDER_TYPE_SELL

            if min_dist > 0:
                if abs(price - norm_sl) < min_dist:
                    logger.warning(f"[MT5] Jarak SL ({abs(price - norm_sl):.{digits}f}) lebih kecil dari stops_level ({min_dist:.{digits}f}). Menyesuaikan SL ke batas aman.")
                    if side == "BUY":
                        norm_sl = round(price - min_dist - (2 * point), digits)
                    else:
                        norm_sl = round(price + min_dist + (2 * point), digits)

                if norm_tp > 0 and abs(price - norm_tp) < min_dist:
                    logger.warning(f"[MT5] Jarak TP ({abs(price - norm_tp):.{digits}f}) lebih kecil dari stops_level ({min_dist:.{digits}f}). Menyesuaikan TP ke batas aman.")
                    if side == "BUY":
                        norm_tp = round(price + min_dist + (2 * point), digits)
                    else:
                        norm_tp = round(price - min_dist - (2 * point), digits)

            # --- SMART CAPITAL ADAPTATION (PENYESUAIAN LOT OTOMATIS SESUAI MODAL) ---
            acc_info = self.mt5.account_info()
            if acc_info is not None:
                if acc_info.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
                    if os.environ.get("ALLOW_LIVE_REAL", "0") != "1":
                        self.last_order_error = "Order ke akun REAL diblokir demi keamanan modal (butuh ALLOW_LIVE_REAL=1)."
                        logger.error(f"[MT5] {self.last_order_error}")
                        return None
                currency = str(getattr(acc_info, "currency", "USD") or "USD").upper()
                if currency in ("CNT", "USDC"):
                    self.last_order_error = f"Order ke akun Cent ({currency}) diblokir: unit belum dinormalisasi."
                    logger.error(f"[MT5] {self.last_order_error}")
                    return None

            free_margin = float(getattr(acc_info, "margin_free", 0.0) or 0.0) if acc_info else 1000.0
            equity = float(getattr(acc_info, "equity", 0.0) or 0.0) if acc_info else 1000.0

            req_margin = self.get_required_margin(symbol, side, norm_lots, price)
            min_margin = self.get_required_margin(symbol, side, vmin, price)

            # Jika sisa margin bebas tidak cukup untuk lot yang diminta
            if free_margin > 0 and req_margin > free_margin * 0.95:
                # Cek apakah cukup untuk lot minimum (vmin):
                if free_margin < min_margin:
                    self.last_order_error = (
                        f"Margin Bebas akun (${free_margin:.2f}) tidak cukup untuk membuka {symbol} "
                        f"(butuh margin minimal ${min_margin:.2f} untuk lot terkecil {vmin}). "
                        f"Tutup posisi terbuka yang ada terlebih dahulu atau gunakan pair Forex (EURUSD/GBPUSD) ber-margin lebih ringan."
                    )
                    logger.warning(f"[MT5] {self.last_order_error}")
                    return None

                # Hitung lot yang masih terjangkau margin bebas
                unit_margin = (req_margin / norm_lots) if norm_lots > 0 else (min_margin / vmin)
                affordable_raw = (free_margin * 0.90) / unit_margin if unit_margin > 0 else vmin
                affordable_lot = round(min(max(math.floor(affordable_raw / step) * step, vmin), norm_lots), 4)

                if affordable_lot < norm_lots:
                    logger.info(
                        f"[MT5 Auto-Lot] Menyesuaikan lot dari {norm_lots} ke {affordable_lot} lot "
                        f"agar sesuai saldo/free margin (${free_margin:.2f}, butuh margin ${self.get_required_margin(symbol, side, affordable_lot, price):.2f})."
                    )
                    norm_lots = affordable_lot

            # Coba eksekusi dengan filling IOC, FOK, lalu RETURN
            for filling in (self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_FOK, self.mt5.ORDER_FILLING_RETURN):
                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": norm_lots,
                    "type": order_type,
                    "price": price,
                    "sl": norm_sl,
                    "tp": norm_tp,
                    "deviation": 20,
                    "magic": self.magic,
                    "comment": re.sub(r'[^a-zA-Z0-9_\-\.\s]', '', str(comment))[:31],
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": filling,
                }
                res = self.mt5.order_send(request)
                if res and res.retcode == self.mt5.TRADE_RETCODE_DONE:
                    self._invalidate_caches()
                    logger.info(f"[MT5] Order #{res.order} Sukses: {side} {norm_lots} lot @ {price}")
                    return Position(
                        ticket=res.order, symbol=symbol, side=side, lots=norm_lots,
                        open_price=price, current_price=price, sl=norm_sl, tp=norm_tp,
                        risk_pips=risk_pips, r_target=1.5,
                        open_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        comment=comment, magic=self.magic,
                    )
                elif res and res.retcode == 10030:  # Unsupported filling mode
                    continue
                else:
                    ret = getattr(res, 'retcode', None)
                    err_msg = res.comment if res else str(self.mt5.last_error())
                    if ret == 10027:
                        err_msg = "Tombol 'Algo Trading' di MT5 belum aktif (berwarna merah). Klik tombol 'Algo Trading' (Ctrl+E) di toolbar atas MT5 agar menjadi HIJAU."
                    elif ret == 10019 or "No money" in str(err_msg) or "not enough money" in str(err_msg).lower():
                        err_msg = (
                            f"Saldo / Margin Bebas tidak mencukupi di broker MT5 (${free_margin:.2f}). "
                            f"Gunakan lot minimum {vmin} lot atau tutup posisi terbuka yang sedang berjalan."
                        )
                    self.last_order_error = err_msg
                    logger.error(f"[MT5] Order Gagal (retcode {ret}): {err_msg}")
                    break
        return None

    def modify_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        with self._lock:
            pos_list = self.mt5.positions_get(ticket=ticket)
            digits = 5
            if pos_list:
                info = self.mt5.symbol_info(pos_list[0].symbol)
                digits = int(getattr(info, "digits", symbol_decimals(pos_list[0].symbol))) if info else symbol_decimals(pos_list[0].symbol)
            request = {
                "action": self.mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": round(float(new_sl), digits) if new_sl and new_sl > 0 else 0.0,
                "tp": round(float(new_tp), digits) if new_tp and new_tp > 0 else 0.0,
            }
            res = self.mt5.order_send(request)
            return res is not None and res.retcode == self.mt5.TRADE_RETCODE_DONE

    def close_trade(self, ticket: int, reason: str = "MANUAL") -> bool:
        with self._lock:
            positions = self.mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            p = positions[0]
            tick = self.mt5.symbol_info_tick(p.symbol)
            if not tick:
                return False
            order_type = self.mt5.ORDER_TYPE_SELL if p.type == self.mt5.POSITION_TYPE_BUY else self.mt5.ORDER_TYPE_BUY
            price = tick.bid if order_type == self.mt5.ORDER_TYPE_SELL else tick.ask

            # Bersihkan karakter & batasi panjang comment maks 24 char (limit MT5 Python API adalah 29 char)
            clean_reason = re.sub(r'[^a-zA-Z0-9_\-\.]', '', str(reason))
            clean_comment = f"Cls {clean_reason}"[:24].strip()

            for filling in (self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_FOK, self.mt5.ORDER_FILLING_RETURN):
                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "position": ticket,
                    "symbol": p.symbol,
                    "volume": p.volume,
                    "type": order_type,
                    "price": price,
                    "deviation": 20,
                    "magic": self.magic,
                    "comment": clean_comment,
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": filling,
                }
                res = self.mt5.order_send(request)
                if res is None or res.retcode != self.mt5.TRADE_RETCODE_DONE:
                    # Fallback jika broker menolak comment kustom: coba tanpa comment
                    request["comment"] = ""
                    res = self.mt5.order_send(request)

                if res is not None and res.retcode == self.mt5.TRADE_RETCODE_DONE:
                    self._invalidate_caches()
                    is_bot = (getattr(p, 'magic', 0) == self.magic)
                    trade_record = {
                        "ticket": ticket,
                        "symbol": p.symbol,
                        "side": "BUY" if p.type == self.mt5.POSITION_TYPE_BUY else "SELL",
                        "lots": p.volume,
                        "open_price": p.price_open,
                        "close_price": price,
                        "close_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "pnl": round(p.profit, 2),
                        "source": "BOT" if is_bot else "MANUAL",
                        "reason": reason,
                    }
                    self.closed_trades.append(trade_record)
                    save_trade_record(trade_record)
                    return True
                elif res and res.retcode == 10030:
                    continue
                else:
                    err_msg = res.comment if res else str(self.mt5.last_error())
                    self.last_order_error = err_msg
                    logger.warning(f"[MT5] Close #{ticket} Gagal dengan filling {filling}: {err_msg}")
        return False

    def get_trade_history(self) -> List[dict]:
        from datetime import timedelta
        now_mono = time.monotonic()
        if now_mono - self._hist_cache_ts < 2.0 and self._hist_cache:
            return list(self._hist_cache)

        reset_info = get_session_reset_info()
        reset_epoch = float(reset_info.get("reset_time", 0))

        res = []
        seen_tickets = set()
        with self._lock:
            try:
                from_date = datetime.now() - timedelta(days=90)
                to_date = datetime.now() + timedelta(days=1)
                deals = self.mt5.history_deals_get(from_date, to_date)
                if deals:
                    by_pos = {}
                    for d in deals:
                        if not d.symbol:
                            continue
                        pid = getattr(d, 'position_id', 0)
                        if not pid:
                            continue
                        if pid not in by_pos:
                            by_pos[pid] = []
                        by_pos[pid].append(d)

                    for pid, d_list in by_pos.items():
                        out_deal = next((d for d in d_list if getattr(d, 'entry', 0) == self.mt5.DEAL_ENTRY_OUT), None)
                        if out_deal:
                            if reset_epoch > 0 and getattr(out_deal, 'time', 0) < reset_epoch:
                                continue
                            in_deal = next((d for d in d_list if getattr(d, 'entry', 0) == self.mt5.DEAL_ENTRY_IN), d_list[0])
                            orig_side = "BUY" if in_deal.type == self.mt5.DEAL_TYPE_BUY else "SELL"
                            pnl = round(out_deal.profit + getattr(out_deal, 'swap', 0) + getattr(out_deal, 'commission', 0), 2)

                            # Identifikasi Eksekutor: BOT vs MANUAL
                            in_magic = getattr(in_deal, 'magic', 0)
                            in_comm = str(getattr(in_deal, 'comment', '') or '').strip().lower()
                            out_comm = str(getattr(out_deal, 'comment', '') or '').strip().lower()
                            is_manual = (
                                ('manual' in in_comm)
                                or ('manual' in out_comm)
                                or (in_magic == 0)
                                or ('client' in in_comm)
                                or ('mobile' in in_comm)
                            )
                            is_bot = (not is_manual) and (
                                (in_magic == self.magic)
                                or any(k in in_comm for k in ('scalp', 'finexalgo', 'ma-fib', 'bot', 'auto'))
                            )
                            source = "BOT" if is_bot else "MANUAL"

                            # Deteksi Alasan Penutupan
                            if "[tp" in out_comm:
                                reason = "TP"
                            elif "[sl" in out_comm:
                                reason = "SL"
                            elif "manual_ui" in out_comm or "close: manual_ui" in out_comm:
                                reason = "MANUAL_UI"
                            elif "manual_close_all" in out_comm:
                                reason = "MANUAL_CLOSE_ALL"
                            elif "manual" in out_comm:
                                reason = "MANUAL"
                            elif getattr(out_deal, 'comment', None):
                                reason = out_deal.comment
                            else:
                                reason = "TP" if pnl >= 0 else "SL"

                            seen_tickets.add(pid)
                            technique = infer_technique_from_text(f"{in_comm} {out_comm} {reason}")
                            res.append({
                                "ticket": pid,
                                "symbol": in_deal.symbol,
                                "side": orig_side,
                                "lots": in_deal.volume,
                                "open_price": in_deal.price,
                                "close_price": out_deal.price,
                                "open_time": datetime.fromtimestamp(in_deal.time).strftime("%Y-%m-%d %H:%M:%S"),
                                "close_time": datetime.fromtimestamp(out_deal.time).strftime("%Y-%m-%d %H:%M:%S"),
                                "close_epoch": int(out_deal.time),
                                "pnl": pnl,
                                "source": source,
                                "reason": reason,
                                "technique": technique,
                            })
            except Exception:
                pass

        # Masukkan posisi lokal yang baru ditutup jika belum terambil di batch deal
        for t in reversed(self.closed_trades):
            if t.get("ticket") not in seen_tickets:
                if reset_epoch > 0 and t.get("close_epoch", 0) < reset_epoch:
                    continue
                if "technique" not in t:
                    t["technique"] = infer_technique_from_text(f"{t.get('comment', '')} {t.get('reason', '')}")
                seen_tickets.add(t.get("ticket"))
                if "close_epoch" not in t:
                    t["close_epoch"] = int(time.time())
                res.append(t)

        # Susun agar riwayat TERBARU SELALU PALING ATAS (close_epoch & ticket descending)
        res.sort(key=lambda x: (int(x.get("close_epoch", 0)), int(x.get("ticket", 0))), reverse=True)
        sync_trade_records_to_file(res)
        self._hist_cache = res[:5000]
        self._hist_cache_ts = now_mono
        return list(self._hist_cache)

    def close_all(self, reason: str = "KILL_SWITCH") -> int:
        with self._lock:
            positions = self.mt5.positions_get()
        if not positions:
            return 0
        count = 0
        for p in positions:
            if self.close_trade(p.ticket, reason):
                count += 1
        return count



class OrderManager:
    """Manajer order terpusat yang mengatur aturan risiko, BE, dan eksekusi."""

    def __init__(self, executor: IExecutor, params: StrategyParams):
        self.executor = executor
        self.params = params
        self.daily_r_pnl: float = 0.0
        self.last_trade_day = datetime.now(timezone.utc).date()
        self.cooldown_remaining: int = 0
        self.logs: List[dict] = []
        self.risk_usd_by_ticket: dict = {}   # ticket -> risiko USD saat entry (untuk hitung R harian)
        self._peak_pnl_by_ticket: dict = {}  # ticket -> float (puncak floating pnl untuk trailing stop)
        self.market_bias_getter: Optional[Any] = None # Callback evaluasi arah pasar by data M1 real-time

    def log(self, message: str, level: str = "INFO"):
        entry = {
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        self.logs.append(entry)
        if len(self.logs) > 500:
            self.logs.pop(0)
        logger.info(f"[{level}] {message}")

    def reset_daily_if_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_trade_day:
            self.daily_r_pnl = 0.0
            self.last_trade_day = today
            self.log(f"Hari baru ({today}): Counter rugi harian direset ke 0R.", "INFO")

    def evaluate_and_execute(self, symbol: str, side: int, sl: float, tp: float, current_price: float, spread_pips: float) -> Optional[Position]:
        self.reset_daily_if_new_day()

        if side == 0:
            return None

        explore = bool(getattr(self.params, "explore_mode", False))

        # 1. Cek Circuit Breaker Rugi Harian (Trader Safety)
        limit_r = self.params.explore_daily_loss_r if explore else self.params.max_daily_loss_r
        if not can_trade_today(self.daily_r_pnl, limit_r):
            self.log(f"CIRCUIT BREAKER: Batas rugi harian (-{limit_r}R) tersentuh ({self.daily_r_pnl:+.2f}R). Trading hari ini dihentikan.", "RISK_ALERT")
            return None

        # 1b. Mode latihan: batasi jumlah posisi terbuka bersamaan
        if explore and len(self.executor.get_positions()) >= self.params.explore_max_open:
            self.log(f"MODE LATIHAN: sudah ada {self.params.explore_max_open} posisi terbuka (batas). Entry baru ditunda.", "INFO")
            return None

        # 2. Cek Cooldown (dilewati di mode latihan; laju entry diatur explore_every_bars)
        if self.cooldown_remaining > 0 and not explore:
            self.log(f"Cooldown aktif ({self.cooldown_remaining} candle tersisa). Sinyal diabaikan.", "INFO")
            return None

        # 3. Cek Spread Guard
        if spread_pips > self.params.max_spread_pips:
            self.log(f"SPREAD GUARD: Spread ({spread_pips:.1f} pip) melebihi batas maksimal ({self.params.max_spread_pips} pip). Batal entry.", "WARNING")
            return None

        # 4. Hitung Jarak SL dan Lot Size
        pip = self.params.pip_size
        risk_dist = abs(current_price - sl)
        reward_dist = abs(tp - current_price)
        risk_pips = risk_dist / pip

        if risk_pips < self.params.min_sl_pips or risk_pips > self.params.max_sl_pips:
            self.log(f"SL FILTER: Jarak SL ({risk_pips:.1f} pip) di luar batas aman [{self.params.min_sl_pips}, {self.params.max_sl_pips}].", "WARNING")
            return None

        rr = reward_dist / risk_dist if risk_dist > 0 else 0
        if rr < self.params.min_rr:
            self.log(f"RR FILTER: Rasio RR ({rr:.2f}) di bawah batas minimum {self.params.min_rr}.", "WARNING")
            return None

        # Hitung ukuran lot berdasarkan risiko % modal + spesifikasi broker live
        acc = self.executor.get_account_info()
        equity = float(acc.get("equity", 0.0) or 0.0)
        if equity <= 0:
            self.log("EKSEKUSI DITOLAK: Informasi ekuitas akun tidak tersedia atau 0 (fail-closed).", "WARNING")
            return None

        specs = self.executor.get_symbol_trade_specs(symbol, pip, current_price)
        pip_val = specs["pip_value"]
        digits = specs["digits"]
        sl = round(float(sl), digits)
        tp = round(float(tp), digits)
        lots = lot_size(
            equity,
            self.params.risk_pct,
            risk_pips,
            pip_val,
            min_lot=specs["min_lot"],
            lot_step=specs["lot_step"],
            max_lot=specs["max_lot"],
        )

        if lots <= 0:
            self.log(f"LOT FILTER: Ukuran lot 0 (ekuitas ${equity:.2f} tidak mencukupi untuk risiko {self.params.risk_pct}% pada SL {risk_pips:.1f} pip). Order dibatalkan.", "WARNING")
            return None

        # 5. Eksekusi Order
        side_str = "BUY" if side == 1 else "SELL"
        pos = self.executor.open_trade(
            symbol=symbol,
            side=side_str,
            lots=lots,
            sl=sl,
            tp=tp,
            risk_pips=risk_pips,
            comment=f"FibBot-{side_str}",
        )

        if pos:
            self.log(f"ENTRY SUKSES: {side_str} {pos.lots} lot @ {current_price:.{digits}f} | SL: {sl:.{digits}f} | TP: {tp:.{digits}f} | Risk: {risk_pips:.1f} pips", "SIGNAL")
            self.risk_usd_by_ticket[pos.ticket] = pos.lots * risk_pips * pip_val
            if not explore:
                self.cooldown_remaining = self.params.cooldown_bars
        return pos

    def execute_scalping(
        self,
        symbol: str,
        side: int,
        sl: float,
        tp: float,
        current_price: float,
        spread_pips: float,
        lot: float = 0.0,
        target_usd: float = 5.0,
        technique: str = "ALL",
    ) -> Optional[Position]:
        """Eksekusi order cepat (Scalping) dengan target cuan spesifik ($2, $5, $10) dan pelacakan teknik."""
        self.reset_daily_if_new_day()
        if side == 0:
            return None

        # 1. Batas Rugi Harian (Circuit Breaker)
        limit_r = getattr(self.params, "explore_daily_loss_r", 10.0) if getattr(self.params, "explore_mode", False) else self.params.max_daily_loss_r
        if not can_trade_today(self.daily_r_pnl, limit_r):
            self.log(f"CIRCUIT BREAKER: Batas rugi harian (-{limit_r}R) tersentuh ({self.daily_r_pnl:+.2f}R). Bot ditahan.", "RISK_ALERT")
            return None

        # 2. Batas Posisi Terbuka Scalping & Multi-Pair Cap
        max_open = int(getattr(self.params, "scalping_max_open", 3))
        open_pos = self.executor.get_positions()
        if len(open_pos) >= max_open:
            return None

        # Batas posisi per simbol (maksimal 1 posisi per pair agar tidak over-exposure)
        max_per_sym = int(getattr(self.params, "scalping_max_per_symbol", 1))
        sym_open = [p for p in open_pos if p.symbol == symbol]
        if len(sym_open) >= max_per_sym:
            return None

        # 3. Cek Spread Guard
        max_sp = float(getattr(self.params, "max_spread_pips", 6.0))
        if spread_pips > max_sp:
            self.log(f"SPREAD GUARD: Spread ({spread_pips:.1f} pip) melebihi batas {max_sp:.1f} pip. Batal entry.", "WARNING")
            return None

        # 4. Tentukan Ukuran Lot
        pip = get_symbol_pip_size(symbol)
        digits = symbol_decimals(symbol)
        specs = self.executor.get_symbol_trade_specs(symbol, pip, current_price)
        min_l, step_l, max_l = specs["min_lot"], specs["lot_step"], specs["max_lot"]
        pip_val = estimate_pip_value(symbol, pip)
        risk_pips = max(2.5, abs(current_price - sl) / pip) if pip > 0 else 10.0

        # 4. Tentukan Ukuran Lot: Mengikuti lot yang diatur pengguna (default 0.1 lot konsisten)
        if lot is None or lot <= 0:
            lot = float(getattr(self.params, "scalping_lot", 0.1) or 0.1)

        # Cek kondisi saldo dan margin akun agar beradaptasi otomatis
        acc = self.executor.get_account_info()
        equity = float(acc.get("equity", 100.0) or 100.0)
        free_margin = float(acc.get("free_margin", equity) or equity)

        # Validasi ukuran lot terhadap batasan broker (tanpa memaksakan secara diam-diam)
        if lot is None or float(lot) <= 0 or float(lot) < min_l:
            self.log(f"LOT FILTER: Ukuran lot ({lot}) di bawah batas minimum broker ({min_l}). Order dibatalkan.", "WARNING")
            return None
        if step_l > 0:
            lot = round(math.floor(float(lot) / step_l + 1e-9) * step_l, 4)
        lot = min(max_l, lot)

        side_str = "BUY" if side == 1 else "SELL"

        # Validasi margin sebelum mengirim order agar tidak tertolak MT5
        if hasattr(self.executor, "get_required_margin"):
            req_m = self.executor.get_required_margin(symbol, side_str, lot, current_price)
            min_m = self.executor.get_required_margin(symbol, side_str, min_l, current_price)
            if free_margin > 0 and free_margin < min_m:
                self.log(
                    f"💰 MARGIN TIDAK CUKUP: Free Margin (${free_margin:.2f}) di bawah syarat margin {symbol} (${min_m:.2f} untuk {min_l} lot). Order ditahan agar tidak ditolak MT5.",
                    "WARNING"
                )
                return None
            elif free_margin > 0 and req_m > free_margin * 0.90:
                self.log(
                    f"💰 MARGIN TIDAK CUKUP: Free Margin (${free_margin:.2f}) tidak mencukupi untuk membuka {lot} lot {symbol} (syarat: ${req_m:.2f}). Order dibatalkan.",
                    "WARNING"
                )
                return None

        pos = self.executor.open_trade(
            symbol=symbol,
            side=side_str,
            lots=lot,
            sl=round(float(sl), digits),
            tp=round(float(tp), digits),
            risk_pips=risk_pips,
            comment=f"Scalp-{technique}-${target_usd:.0f}",
        )
        if pos:
            self.log(
                f"🚀 ENTRY SCALPING [{technique}] {side_str} {pos.lots} lot @ {current_price:.{digits}f} | TP: {tp:.{digits}f} (Target +${target_usd:.2f}) | SL: {sl:.{digits}f}",
                "SIGNAL",
            )
            self.risk_usd_by_ticket[pos.ticket] = pos.lots * risk_pips * pip_val
        return pos

    def register_closed_trade(self, t: dict) -> None:
        """Catat hasil trade yang sudah tertutup ke penghitung R harian (dasar circuit breaker)."""
        tk = t.get("ticket")
        risk_usd = self.risk_usd_by_ticket.pop(tk, None)
        if risk_usd is None:
            return
        r = t.get("r")
        if r is None:
            r = float(t.get("pnl", 0.0)) / risk_usd if risk_usd > 0 else 0.0
        self.reset_daily_if_new_day()
        self.daily_r_pnl += float(r)
        self.log(f"Trade #{tk} tutup: {float(r):+.2f}R | total hari ini {self.daily_r_pnl:+.2f}R", "INFO")

    def manage_active_positions(self) -> None:
        """
        Monitor proteksi profit dinamis & eksekusi anti-balik minus:
        1. Continuous Peak Tracking: pantau puncak profit tertinggi tiap detik.
        2. Dynamic Trailing Locking: kunci profit berjenjang (35% - 70% dari puncak).
        3. Break-Even Activation: kunci BE + fees jika harga bergerak searah target.
        4. Emergency Catastrophic Cap (-$5.00): batas pengaman rugi darurat.
        """
        positions = self.executor.get_positions()
        target_usd = float(getattr(self.params, "target_profit_usd", 5.0) or 5.0)
        configured_loss_usd = float(getattr(self.params, "target_loss_usd", 5.0) or 5.0)
        max_loss_usd = min(configured_loss_usd, round(target_usd * 1.0, 2)) if target_usd > 0 else configured_loss_usd

        # Bersihkan tiket yang sudah tertutup dari memori puncak profit
        active_tickets = set(p.ticket for p in positions)
        for t in list(self._peak_pnl_by_ticket.keys()):
            if t not in active_tickets:
                self._peak_pnl_by_ticket.pop(t, None)

        for p in positions:
            digits = symbol_decimals(p.symbol)
            pip = get_symbol_pip_size(p.symbol)
            is_gold = "XAU" in p.symbol.upper() or "GOLD" in p.symbol.upper()
            is_macro_sniper = (
                "CPI_SNIPER" in str(getattr(p, "comment", "")).upper() or
                "PPI_SNIPER" in str(getattr(p, "comment", "")).upper() or
                getattr(p, "technique", "") in ("CPI_SNIPER", "PPI_SNIPER")
            )

            # 0. CONTINUOUS PEAK TRACKING: Catat selalu puncak floating profit tertinggi
            prev_peak = self._peak_pnl_by_ticket.get(p.ticket, 0.0)
            cur_peak = max(prev_peak, float(p.floating_pnl))
            self._peak_pnl_by_ticket[p.ticket] = cur_peak
            p.peak_pnl = round(cur_peak, 2)
            peak = float(p.peak_pnl)

            cur_pips = 0.0
            if pip > 0:
                cur_pips = (p.current_price - p.open_price) / pip if p.side == "BUY" else (p.open_price - p.current_price) / pip

            # 1. SMART DYNAMIC PROFIT TRAILING
            locked_profit_usd = 0.0
            if not is_macro_sniper:
                if peak >= 5.00:
                    locked_profit_usd = max(3.50, round(peak * 0.70, 2))
                elif peak >= 2.50:
                    locked_profit_usd = max(1.50, round(peak - 1.00, 2))
                elif peak >= 1.00:
                    locked_profit_usd = 0.35
            else:
                if peak >= 6.00:
                    locked_profit_usd = max(4.50, round(peak * 0.80, 2))
                elif peak >= 3.50:
                    locked_profit_usd = max(2.50, round(peak - 1.00, 2))

            # 2. DUAL-LAYER ENFORCEMENT:
            # LAYER A: UPDATE STOP LOSS DI BROKER MT5 SECARA MERAYAP (TRAILING SL BROKER)
            if locked_profit_usd > 0 and pip > 0:
                pip_val = estimate_pip_value(p.symbol, pip)
                dollar_per_pip = (pip_val * p.lots) if (pip_val > 0 and p.lots > 0) else 0.10
                target_locked_pips = locked_profit_usd / dollar_per_pip if dollar_per_pip > 0 else 0.0
                min_buffer_pips = 10.0 if is_gold else 3.0

                if p.side == "BUY":
                    target_sl = round(p.open_price + (target_locked_pips * pip), digits)
                    # Modifikasi jika lebih menguntungkan minimal 1.0 pip dari SL lama dan berjarak aman dari running price
                    if (target_sl > p.sl + (1.0 * pip)) and (target_sl < p.current_price - (min_buffer_pips * pip)):
                        if self.executor.modify_sl_tp(p.ticket, target_sl, p.tp):
                            old_sl = p.sl
                            p.sl = target_sl
                            p.be_active = True
                            self.log(
                                f"🛡️ TRAILING SL NAIK (BUY): Order #{p.ticket} ({p.symbol}) SL dinaikkan dari {old_sl:.{digits}f} ke {target_sl:.{digits}f} (+${locked_profit_usd:.2f} terkunci di broker!). Puncak: +${peak:.2f}",
                                "INFO",
                            )
                else: # SELL
                    target_sl = round(p.open_price - (target_locked_pips * pip), digits)
                    # Untuk SELL: SL lebih rendah adalah lebih menguntungkan
                    if (p.sl <= 0 or target_sl < p.sl - (1.0 * pip)) and (target_sl > p.current_price + (min_buffer_pips * pip)):
                        if self.executor.modify_sl_tp(p.ticket, target_sl, p.tp):
                            old_sl = p.sl
                            p.sl = target_sl
                            p.be_active = True
                            self.log(
                                f"🛡️ TRAILING SL TURUN (SELL): Order #{p.ticket} ({p.symbol}) SL diturunkan dari {old_sl:.{digits}f} ke {target_sl:.{digits}f} (+${locked_profit_usd:.2f} terkunci di broker!). Puncak: +${peak:.2f}",
                                "INFO",
                            )

            # LAYER B: ENGINE ACTIVE PROFIT LOCK GUARD (TUTUP DI PASAR JIKA MENUKIK MENEMBUS LEVEL KUNCI)
            if locked_profit_usd > 0 and p.floating_pnl <= locked_profit_usd:
                ok = self.executor.close_trade(p.ticket, reason=f"PROFIT_LOCK_{locked_profit_usd:.2f}_PEAK_{peak:.2f}")
                if ok:
                    self.log(
                        f"💰 PROFIT LOCK DIAMANKAN: Order #{p.ticket} ({p.symbol} {p.side}) diamankan di +${p.floating_pnl:.2f} (Kunci Pengaman: +${locked_profit_usd:.2f} | Puncak: +${peak:.2f}) sebelum tergerus! Cuan diamankan ke saldo.",
                        "INFO",
                    )
                    continue

            # 3. UNLIMITED DYNAMIC TP EXPANSION (TP JANGAN DIBATASI SELAMA MOMENTUM/DATA BERJALAN)
            if p.tp > 0 and pip > 0:
                dist_to_tp_pips = abs(p.tp - p.current_price) / pip
                if (p.floating_pnl >= 3.50) and (dist_to_tp_pips <= 15.0):
                    expansion_pips = 40.0 if not is_gold else 60.0
                    new_tp = round(p.current_price + (expansion_pips * pip) if p.side == "BUY" else p.current_price - (expansion_pips * pip), digits)
                    if self.executor.modify_sl_tp(p.ticket, p.sl, new_tp):
                        old_tp = p.tp
                        p.tp = new_tp
                        self.log(
                            f"🚀 UNLIMITED RUNNER TP DIPERPANJANG: Order #{p.ticket} ({p.symbol}) TP digeser dari {old_tp:.{digits}f} ke {new_tp:.{digits}f} (+{expansion_pips:.0f} pip lebih jauh). Membiarkan cuan berlari tanpa batas!",
                            "INFO",
                        )

            # 4. PENGAMAN BATAS RUGI DARURAT PROPORSIONAL ($)
            if not is_macro_sniper and max_loss_usd > 0 and p.floating_pnl <= -max_loss_usd:
                ok = self.executor.close_trade(p.ticket, reason=f"STOP_LOSS_{max_loss_usd:.2f}")
                if ok:
                    self.log(
                        f"⚠️ CUT LOSS DARURAT: Order #{p.ticket} ({p.symbol}) ditutup untuk batasi rugi pada ${p.floating_pnl:.2f} (Max: -${max_loss_usd:.2f})",
                        "RISK_ALERT",
                    )
                    continue
