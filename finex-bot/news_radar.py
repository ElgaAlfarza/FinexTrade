"""Modul Radar Berita & Kalender Ekonomi Finansial (High-Impact News Guard).

Memantau rilis data ekonomi utama penggerak pasar:
1. CPI (Consumer Price Index / Inflasi)
2. NFP (Non-Farm Payrolls / Ketenagakerjaan AS)
3. GDP (Gross Domestic Product / PDB)
4. Suku Bunga & Kebijakan Bank Sentral (FOMC / The Fed, ECB, BOE)
5. PPI, Klaim Pengangguran (Unemployment Claims), & Retail Sales

Fitur Proteksi Modal (Institutional News Guard):
- Auto Circuit Breaker: Menahan open posisi baru 15 menit sebelum & sesudah rilis High-Impact
  guna melindungi akun dari lonjakan spread mendadak, slippage, dan volatilitas liar.
- Komparasi Konsensus: Forecast vs Previous vs Aktual & Aturan Reaksi Arah Pasar.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("finex.news_radar")

CALENDAR_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_FILE = r"d:\BigBoz\FinexTrade\finex-bot\data\economic_calendar_cache.json"

# Panduan Reaksi Pasar terhadap Indikator Ekonomi Utama
MACRO_REACTION_GUIDE = {
    "CPI": {
        "name": "CPI (Consumer Price Index / Inflasi)",
        "frequency": "Bulanan (Sekitar tanggal 10-14)",
        "higher_reaction": "Inflasi Tinggi -> The Fed cenderung Hawkish (Tahan/Naikkan Bunga) -> USD MENGUAT, Emas (XAUUSD) MELEMAH / Jatuh.",
        "lower_reaction": "Inflasi Mereda -> The Fed berpeluang Dovish (Pangkas Bunga) -> USD MELEMAH, Emas (XAUUSD) MENGUAT / Meroket.",
        "volatility": "Sangat Tinggi (±50 - 100 pips dalam hitungan detik)."
    },
    "NFP": {
        "name": "NFP (Non-Farm Payrolls / Tenaga Kerja AS)",
        "frequency": "Jumat pertama setiap awal bulan (Departemen Tenaga Kerja AS)",
        "higher_reaction": "Lapangan Kerja Kuat -> Ekonomi Tangguh -> USD MENGUAT, Emas (XAUUSD) MELEMAH.",
        "lower_reaction": "Lapangan Kerja Lesu -> Kekhawatiran Resesi -> USD MELEMAH, Emas (XAUUSD) MENGUAT.",
        "volatility": "Ekstrem (±60 - 150 pips)."
    },
    "GDP": {
        "name": "GDP (Gross Domestic Product / PDB)",
        "frequency": "Kuartalan (Kuartal I, II, III, IV)",
        "higher_reaction": "Pertumbuhan Ekonomi Kuat -> Permintaan Mata Uang Meningkat -> USD/EUR MENGUAT.",
        "lower_reaction": "Ekonomi Melambat -> Prospek Melemah -> USD/EUR MELEMAH.",
        "volatility": "Tinggi (±30 - 60 pips)."
    },
    "INTEREST_RATE": {
        "name": "Suku Bunga & Pernyataan Bank Sentral (The Fed, ECB, BOE)",
        "frequency": "8 kali setahun (Rapat FOMC / Dewan Gubernur ECB)",
        "higher_reaction": "Kenaikan Suku Bunga (Hike) -> Yield Dolar Meningkat -> USD MENGUAT DERAS, Emas TERTEKAN.",
        "lower_reaction": "Pemotongan Suku Bunga (Cut) -> Yield Dolar Turun -> USD MELEMAH, Emas TERBANG TINGGI.",
        "volatility": "Super Ekstrem (Trend Reversal Jangka Panjang)."
    }
}


class EconomicNewsRadar:
    """Radar pemantau berita ekonomi dan kalender fundamental global real-time."""

    def __init__(self, cache_duration_sec: int = 900):
        self.cache_duration_sec = cache_duration_sec
        self._cached_events: List[Dict[str, Any]] = []
        self._last_fetch_ts: float = 0.0
        self.news_shield_enabled: bool = True  # Proteksi auto-pause saat high-impact news
        self.shield_buffer_minutes: int = 15   # Jeda 15 menit sebelum & sesudah berita
        self._load_local_cache()

    def _load_local_cache(self) -> None:
        """Muat data kalender dari file lokal jika tersedia."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._cached_events = data.get("events", [])
                    self._last_fetch_ts = data.get("fetched_at", 0.0)
            except Exception as e:
                logger.debug(f"Gagal baca cache kalender: {e}")

    def _save_local_cache(self) -> None:
        """Simpan kalender ke cache lokal."""
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "fetched_at": self._last_fetch_ts,
                    "updated_str": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "events": self._cached_events
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"Gagal simpan cache kalender: {e}")

    def fetch_calendar_events(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Ambil data kalender ekonomi real-time dari Forex Factory feed."""
        now_ts = time.time()
        if not force_refresh and self._cached_events and (now_ts - self._last_fetch_ts) < self.cache_duration_sec:
            return self._cached_events

        req = urllib.request.Request(
            CALENDAR_FEED_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw_data = json.loads(resp.read().decode("utf-8"))
                parsed_events = []
                for ev in raw_data:
                    c = str(ev.get("country", "")).upper()
                    impact = str(ev.get("impact", "")).capitalize()
                    title = str(ev.get("title", ""))
                    date_str = str(ev.get("date", ""))

                    # Filter mata uang relevan dengan portofolio Finex (USD, EUR, GBP)
                    if c in ("USD", "EUR", "GBP") and impact in ("High", "Medium", "Low"):
                        # Analisis kategori berita
                        cat = "OTHER"
                        u_title = title.upper()
                        if "CPI" in u_title or "INFLATION" in u_title:
                            cat = "CPI"
                        elif "PAYROLL" in u_title or "NFP" in u_title or "EMPLOYMENT" in u_title:
                            cat = "NFP"
                        elif "GDP" in u_title:
                            cat = "GDP"
                        elif "RATE" in u_title or "FOMC" in u_title or "FED" in u_title or "ECB" in u_title or "POWELL" in u_title or "LAGARDE" in u_title:
                            cat = "INTEREST_RATE"
                        elif "PPI" in u_title:
                            cat = "PPI"
                        elif "CLAIM" in u_title:
                            cat = "CLAIMS"

                        # Estimasi dampak ke XAUUSD
                        impact_note = ""
                        if cat == "CPI":
                            impact_note = "Actual > Forecast: USD Menguat, Emas Melemah. Actual < Forecast: Emas Menguat."
                        elif cat == "NFP":
                            impact_note = "NFP Kuat (> Forecast): USD Naik, Emas Jatuh. NFP Lemah: Emas Naik."
                        elif cat == "INTEREST_RATE":
                            impact_note = "Hawkish / Suku Bunga Naik: Emas Tertekan. Dovish / Pangkas Bunga: Emas Terbang."
                        elif cat == "GDP":
                            impact_note = "GDP Kuat: Penguatan mata uang terkait."

                        parsed_events.append({
                            "title": title,
                            "country": c,
                            "date": date_str,
                            "impact": impact,
                            "forecast": str(ev.get("forecast", "") or "-"),
                            "previous": str(ev.get("previous", "") or "-"),
                            "category": cat,
                            "impact_note": impact_note,
                        })

                if parsed_events:
                    self._cached_events = parsed_events
                    self._last_fetch_ts = now_ts
                    self._save_local_cache()
                    logger.info(f"Kalender berita terupdate: {len(parsed_events)} event finansial.")
                    return parsed_events
        except Exception as e:
            logger.warning(f"Gagal mengambil kalender langsung dari internet ({e}). Menggunakan data cache.")

        return self._cached_events

    def get_news_shield_status(self) -> Dict[str, Any]:
        """Periksa apakah saat ini sedang dalam zona bahaya rilis berita High-Impact."""
        events = self.fetch_calendar_events()
        now_utc = datetime.now(timezone.utc)
        
        is_shield_active = False
        active_event: Optional[Dict[str, Any]] = None
        closest_event: Optional[Dict[str, Any]] = None
        min_diff_sec = float("inf")

        for ev in events:
            if ev.get("impact") != "High":
                continue
            date_str = ev.get("date", "")
            try:
                ev_time = datetime.fromisoformat(date_str)
                # Standarisasi ke UTC
                diff_sec = (ev_time - now_utc).total_seconds()
                abs_diff = abs(diff_sec)

                if abs_diff < min_diff_sec:
                    min_diff_sec = abs_diff
                    closest_event = ev
                    closest_event["diff_minutes"] = round(diff_sec / 60.0, 1)

                # Cek jika dalam jendela bahaya: 15 menit sebelum hingga 15 menit setelah rilis
                buffer_sec = self.shield_buffer_minutes * 60
                if -buffer_sec <= diff_sec <= buffer_sec:
                    is_shield_active = True
                    active_event = ev
                    active_event["diff_minutes"] = round(diff_sec / 60.0, 1)
                    break
            except Exception:
                continue

        # Status text
        if is_shield_active and active_event:
            status = "NEWS_SHIELD_ACTIVE"
            diff_m = active_event.get("diff_minutes", 0)
            if diff_m > 0:
                warning = f"🚨 NEWS SHIELD AKTIF: [{active_event['country']}] {active_event['title']} rilis dalam {diff_m:.0f} menit! Menahan order guna proteksi lonjakan spread & slippage."
            else:
                warning = f"🚨 NEWS SHIELD AKTIF: [{active_event['country']}] {active_event['title']} baru saja dirilis ({abs(diff_m):.0f}m lalu)! Menunggu volatilitas mereda."
        else:
            status = "SAFE"
            if closest_event:
                diff_m = closest_event.get("diff_minutes", 0)
                if diff_m > 0:
                    warning = f"🟢 Kondisi Pasar Aman: High-Impact berikutnya [{closest_event['country']}] {closest_event['title']} dalam {diff_m/60.0:.1f} jam ({diff_m:.0f} menit)."
                else:
                    warning = f"🟢 Kondisi Pasar Aman: Tidak ada rilis data High-Impact dekat."
            else:
                warning = "🟢 Kondisi Pasar Aman: Kalender terpantau normal."

        return {
            "is_shield_active": is_shield_active and self.news_shield_enabled,
            "shield_enabled": self.news_shield_enabled,
            "status": status,
            "warning": warning,
            "active_event": active_event,
            "closest_event": closest_event,
            "checked_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Kompilasi ringkasan kalender berita lengkap untuk antarmuka dashboard dan bot."""
        events = self.fetch_calendar_events()
        shield = self.get_news_shield_status()
        
        # Pisahkan High vs Medium vs Low
        high_events = [e for e in events if e.get("impact") == "High"]
        med_events = [e for e in events if e.get("impact") == "Medium"]

        return {
            "shield": shield,
            "total_events": len(events),
            "high_impact_count": len(high_events),
            "medium_impact_count": len(med_events),
            "upcoming_high_impact": high_events[:8],
            "upcoming_medium_impact": med_events[:6],
            "macro_reaction_guide": MACRO_REACTION_GUIDE,
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }


# Singleton Global
_news_radar_instance: Optional[EconomicNewsRadar] = None

def get_news_radar() -> EconomicNewsRadar:
    global _news_radar_instance
    if _news_radar_instance is None:
        _news_radar_instance = EconomicNewsRadar()
    return _news_radar_instance
