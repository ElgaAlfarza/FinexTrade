"""Modul Jurnal Trading Otomatis (Trade Journaling & Post-Mortem Evaluator).

Mengimplementasikan 3 Pilar Utama Trader Profesional:
1. Stop Loss Tempat Merubah View:
   - SL adalah titik pembatalan struktur (Invalidation Level).
   - Ketika SL tersentuh, struktur tren dianggap rusak, view langsung berubah ke WAIT / REVERSAL,
     dan dilarang re-entry searah sebelum terkonfirmasi struktur baru.
2. Scalping Mempunyai Zona:
   - Scalping wajib mempunyai zona entry yang jelas (Dynamic Retest Golden Pocket EMA 8/21).
   - Dilarang entry di No Man's Land (tengah pergerakan tanpa zona).
3. Jurnaling Hasil Trading:
   - Pencatatan otomatis setiap posisi tertutup (Win, Loss, BE, Trailing).
   - Menghitung Grade Kedisiplinan (A+, A, B, C) & Insight Refleksi Post-Mortem.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

JOURNAL_FILE = os.path.join(os.path.dirname(__file__), "data", "trade_journal.json")
_lock = threading.Lock()


def evaluate_trade_discipline(pnl: float, reason: str, side: str, open_p: float, close_p: float, entry_in_zone: bool = True) -> tuple[str, str, str, bool]:
    """Evaluasi kedisiplinan trading, status outcome, view shifted, dan insight post-mortem."""
    reason_u = str(reason).upper()
    is_sl = ("SL" in reason_u or "STOP_LOSS" in reason_u or pnl < -0.1)
    is_tp = ("TP" in reason_u or "TAKE_PROFIT" in reason_u or pnl >= 2.0)
    is_be = (pnl >= 0.0 and pnl < 1.0 and ("BE" in reason_u or "SL" in reason_u))

    view_shifted = False
    if is_be:
        outcome = "BREAK_EVEN"
        grade = "A" if entry_in_zone else "B"
        insight = f"Posisi {side} berhasil diamankan via Break-Even (+${pnl:.2f}). Modal 100% terlindungi saat harga berbalik."
    elif is_tp:
        outcome = "WIN_TP"
        grade = "A+" if entry_in_zone else "A"
        insight = f"Target Profit tercapai (+${pnl:.2f})! Eksekusi patuh pada zona scalping & RR positif."
    elif is_sl:
        outcome = "LOSS_SL"
        view_shifted = True
        grade = "B" if entry_in_zone else "C"
        insight = (
            f"Stop Loss tersentuh di {close_p:.2f} (-${abs(pnl):.2f}). "
            f"Level invalidasi struktur terkonfirmasi rusak! View {side} otomatis di-reset ke WAIT guna memutus dendam pasar."
        )
    else:
        outcome = "CLOSED"
        grade = "B"
        insight = f"Posisi {side} ditutup pada hasil ${pnl:.2f} ({reason})."

    return outcome, grade, insight, view_shifted


class TradeJournal:
    def __init__(self, filepath: str = JOURNAL_FILE):
        self.filepath = filepath
        self.entries: List[dict] = []
        self.current_market_view: dict = {
            "bias": "NETRAL",
            "last_invalidation_level": 0.0,
            "status": "VIEW NORMAL",
            "reason": "Pasar mematuhi zona scalping teratur.",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.load()

    def load(self) -> None:
        with _lock:
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.entries = data.get("entries", [])
                        self.current_market_view = data.get("current_market_view", self.current_market_view)
                except Exception:
                    self.entries = []

    def save(self) -> None:
        with _lock:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            payload = {
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_trades": len(self.entries),
                "current_market_view": self.current_market_view,
                "entries": self.entries[:250],  # simpan hingga 250 riwayat jurnal terbaru
            }
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    def record_trade(
        self,
        ticket: int,
        symbol: str,
        side: str,
        open_time: str,
        close_time: str,
        open_price: float,
        close_price: float,
        pnl: float,
        reason: str,
        technique: str = "EMA_TREND",
        entry_in_zone: bool = True,
        zone_description: str = "Dynamic Support/Resistance EMA 8/21 Pocket",
    ) -> dict:
        """Catat transaksi yang baru saja tertutup ke dalam jurnal kuantitatif."""
        ticket_str = str(ticket)
        # Hindari duplikasi tiket
        for e in self.entries:
            if str(e.get("ticket")) == ticket_str:
                return e

        outcome, grade, insight, view_shifted = evaluate_trade_discipline(
            pnl=pnl,
            reason=reason,
            side=side,
            open_p=open_price,
            close_p=close_price,
            entry_in_zone=entry_in_zone,
        )

        # 1. PILAR 1: Stop Loss Tempat Merubah View
        if view_shifted:
            self.current_market_view = {
                "bias": f"VIEW_SHIFTED_DARI_{side}",
                "status": "STRUKTUR RUSAK (WAIT)",
                "last_invalidation_level": round(close_price, 2),
                "reason": f"Stop Loss #{ticket_str} ({side}) tersentuh di {close_price:.2f}. Struktur invalid! Dilarang re-entry {side} sebelum swing baru.",
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        entry = {
            "ticket": ticket_str,
            "symbol": str(symbol).upper(),
            "side": str(side).upper(),
            "technique": technique,
            "open_time": str(open_time),
            "close_time": str(close_time),
            "open_price": round(float(open_price), 2),
            "close_price": round(float(close_price), 2),
            "pnl": round(float(pnl), 2),
            "outcome": outcome,
            "grade": grade,
            "entry_in_zone": entry_in_zone,
            "zone_description": zone_description,
            "view_shifted": view_shifted,
            "insight": insight,
            "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.entries.insert(0, entry)
        self.save()
        return entry

    def sync_from_trade_records(self, csv_path: str) -> int:
        """Sinkronisasi data riwayat transaksi lama ke dalam jurnal jika belum tercatat."""
        if not os.path.exists(csv_path):
            return 0
        import csv
        existing_tickets = set(str(e.get("ticket")) for e in self.entries)
        new_count = 0

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            for r in reversed(rows):
                tk = str(r.get("ticket", "")).strip()
                if not tk or tk in existing_tickets:
                    continue
                sym = str(r.get("symbol", "XAUUSD")).strip()
                side = str(r.get("side", "BUY")).strip()
                pnl = float(r.get("pnl", 0.0) or r.get("profit", 0.0) or 0.0)
                open_p = float(r.get("open_price", 0.0) or 0.0)
                close_p = float(r.get("close_price", 0.0) or 0.0)
                reason = str(r.get("reason", "SL" if pnl < 0 else "TP"))
                o_time = str(r.get("open_time", ""))
                c_time = str(r.get("close_time", ""))
                tech = str(r.get("technique", "EMA_TREND"))

                self.record_trade(
                    ticket=tk,
                    symbol=sym,
                    side=side,
                    open_time=o_time,
                    close_time=c_time,
                    open_price=open_p,
                    close_price=close_p,
                    pnl=pnl,
                    reason=reason,
                    technique=tech,
                    entry_in_zone=True,
                    zone_description="Dynamic Pocket EMA 8/21",
                )
                existing_tickets.add(tk)
                new_count += 1
        except Exception:
            pass

        return new_count

    def get_summary(self) -> dict:
        total = len(self.entries)
        if total == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "be_rate": 0.0,
                "loss_rate": 0.0,
                "zone_adherence_pct": 100.0,
                "total_pnl": 0.0,
                "current_market_view": self.current_market_view,
            }

        wins = sum(1 for e in self.entries if e.get("outcome") in ("WIN_TP", "TRAILING"))
        bes = sum(1 for e in self.entries if e.get("outcome") == "BREAK_EVEN")
        losses = sum(1 for e in self.entries if e.get("outcome") == "LOSS_SL")
        in_zone = sum(1 for e in self.entries if e.get("entry_in_zone", True))
        tot_pnl = sum(float(e.get("pnl", 0.0)) for e in self.entries)

        return {
            "total_trades": total,
            "wins": wins,
            "bes": bes,
            "losses": losses,
            "win_rate": round(wins / total * 100.0, 1),
            "be_rate": round(bes / total * 100.0, 1),
            "risk_free_rate": round((wins + bes) / total * 100.0, 1),
            "loss_rate": round(losses / total * 100.0, 1),
            "zone_adherence_pct": round(in_zone / total * 100.0, 1),
            "total_pnl": round(tot_pnl, 2),
            "current_market_view": self.current_market_view,
        }


# Singleton Global Journal
_journal_instance: Optional[TradeJournal] = None


def get_trade_journal() -> TradeJournal:
    global _journal_instance
    if _journal_instance is None:
        _journal_instance = TradeJournal()
        records_csv = os.path.join(os.path.dirname(__file__), "data", "trade_records.csv")
        if os.path.exists(records_csv):
            _journal_instance.sync_from_trade_records(records_csv)
    return _journal_instance
