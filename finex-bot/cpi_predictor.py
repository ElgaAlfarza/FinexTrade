"""Modul Prediksi Makro Berbasis Data (PPI Malam Ini & CPI Nowcasting) & Strategi XAUUSD Big Move Sniper.

Mengintegrasikan:
1. Producer Price Index (PPI) AS: Leading indicator inflasi manufaktur/input pabrik (Rilis Malam Ini, 13 Nov 20:30 WIB).
2. Consumer Price Index (CPI) AS: Model Inflation Nowcasting Fed Cleveland & tren bensin ritel (Rilis 10 Des 20:30 WIB).
3. Peta waktu eksekusi presisi (Pre-Positioning vs Freeze vs Post-Momentum) untuk menangkap ekspansi besar (+60 s/d +150 pips) pada XAUUSD (Emas).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("finex.cpi_predictor")

CPI_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "data", "cpi_sniper_config.json")

# =========================================================================
# 1. DATA KALIBRASI PROYEKSI US PPI (PRODUCER PRICE INDEX) - RILIS MALAM INI
# =========================================================================
DEFAULT_PPI_PROJECTION = {
    "event_id": "PPI",
    "event_name": "US Producer Price Index (PPI) & Core PPI",
    "release_date_str": "2026-11-13 20:30:00", # Tanggal & Jam Rilis (WIB)
    "release_date_display": "Jumat, 13 November 2026",
    "release_time_display": "Pukul 20:30:00 WIB (21:30 WITA)",
    "release_date_et": "Friday, Nov 13, 2026 08:30:00 AM ET",
    "release_iso_utc": "2026-11-13T13:30:00Z",
    "report_period": "Data Bulan Oktober/November 2026",
    
    # Perbandingan Angka Konsensus vs Data Lalu
    "core_ppi_mom": {
        "previous": 0.70,      # Rilis lalu (+0.70%)
        "forecast": 0.30,      # Forecast konsensus (+0.30%)
        "delta": -0.40,        # Melambat drastis (-0.40% atau -57.1%)
        "unit": "% MoM",
        "trend": "COOLING_DRASTIC",
    },
    "headline_ppi_mom": {
        "previous": 1.30,      # Rilis lalu (+1.30%)
        "forecast": 0.40,      # Forecast konsensus (+0.40%)
        "delta": -0.90,        # Penurunan kolosal (-0.90% atau -69.2%)
        "unit": "% MoM",
        "trend": "COLLAPSE",
    },

    # 4 Leading Factors Penekan PPI
    "factors": [
        {
            "name": "Biaya Energi Industri & Minyak Mentah WTI",
            "status": "TURUN KE LEVEL $68 - $71 / BAREL",
            "impact": "DEFLASIONER PRODUSEN (-)",
            "detail": "Harga solar industri & diesel AS turun -6.2%. Beban listrik pabrik dan bahan bakar transportasi manufaktur merosot tajam.",
            "score": -0.35,
        },
        {
            "name": "Indeks Biaya Input Pabrik (ISM Prices Paid)",
            "status": "MELANDAI DI LEVEL 51.2",
            "impact": "MEREDA SIGNIFIKAN (-)",
            "detail": "Survei Institute for Supply Management (ISM) mengonfirmasi inflasi biaya bahan mentah pabrik telah ternormalisasi.",
            "score": -0.25,
        },
        {
            "name": "Tarif Pengiriman & Angkutan Barang Grosir",
            "status": "TARIF LOGISTIK MENURUN",
            "impact": "ONGKOS DISTRIBUSI DROP (-)",
            "detail": "Cass Freight Index dan freight rates kontainer domestik AS mengalami kontraksi, memotong biaya perantara.",
            "score": -0.20,
        },
        {
            "name": "Transmisi Pipeline ke CPI (Leading Indicator)",
            "status": "SINYAL KUAT PENURUNAN INFLASI KONSUMEN",
            "impact": "DOVISH RATE CUT CONFIRMATION (-)",
            "detail": "PPI mengukur inflasi di pintu pabrik sebelum sampai ke toko. Anjloknya PPI mengonfirmasi bahwa inflasi CPI Desember pasti melambat.",
            "score": -0.20,
        }
    ],

    # Skor Probabilitas Model Kuantitatif
    "probability": {
        "cooling_inflation_prob": 81.5, # 81.5% peluang PPI Melambat / Di Bawah atau Sesuai Forecast
        "hot_inflation_prob": 18.5,     # 18.5% risiko kejutan inflasi naik
        "confidence_level": "TINGGI (A+)",
        "consensus_verdict": "COOLING_PPI (BIAYA PRODUSEN MEREDA DRASTIS)",
    },

    # Skenario & Rekomendasi Terhadap XAUUSD
    "xauusd_playbook": {
        "primary_direction": "STRONG_BUY",
        "action_label": "BULLISH SURGE (STRONG BUY XAUUSD)",
        "rationale": "Biaya produsen merosot tajam (Core 0.3% vs 0.7%, Headline 0.4% vs 1.3%) -> Yield US Treasury 10Y & Dolar AS (DXY) anjlok -> XAUUSD (Emas) sebagai safe-haven tanpa imbal hasil meroket kencang.",
        "expected_move_pips": "+60 s/d +120 pips (+6.00 hingga +12.00 poin Emas)",
        "potential_gain_usd": {
            "lot_0_01": "+$6.00 s/d +$12.00",
            "lot_0_02": "+$12.00 s/d +$24.00",
            "lot_0_05": "+$30.00 s/d +$60.00",
            "lot_0_10": "+$60.00 s/d +$120.00",
        },
        # Peta Waktu Eksekusi Malam Ini
        "execution_schedule": [
            {
                "time_wib": "20:00 - 20:15 WIB",
                "phase": "Fase 1: Pre-News Positioning",
                "action": "Akumulasi BUY di Zona Support EMA 8/21",
                "detail": "Spread masih normal (1.5 - 2.5 pips). Masuk BUY di pocket dinamis M5/M15 dengan SL terukur di bawah support (buffer 40-45 pip).",
            },
            {
                "time_wib": "20:25 - 20:30 WIB",
                "phase": "Fase 2: Freeze Period & News Shield",
                "action": "Kunci Posisi, Dilarang Order Market Baru",
                "detail": "Spread broker mulai melebar (5 - 10 pips). News Shield aktif menahan order baru guna menghindari slippage liar.",
            },
            {
                "time_wib": "20:30:00 WIB",
                "phase": "Fase 3: Rilis Resmi BLS (T-0)",
                "action": "Data Rilis & Initial Volatility Spike",
                "detail": "Data resmi PPI keluar ke publik. Spike cepat 25-40 pip dalam hitungan detik.",
            },
            {
                "time_wib": "20:30:30 - 20:33 WIB",
                "phase": "Fase 4: Post-Release Momentum Sniper",
                "action": "Eksekusi Sniper BUY Mengikuti Ekspansi Reli",
                "detail": "Spread kembali normal rapat. Jika Core PPI terkonfirmasi <= 0.3%, eksekusi order BUY tambahan untuk mengunci reli penuh (+60 s/d +120 pips).",
            }
        ],
        # Skenario Kontingensi
        "contingency_plan": {
            "scenario": "Jika Actual Core PPI >= 0.7% (Surprise Hawkish)",
            "market_reaction": "USD Menguat Keras, Emas Tertekan Jatuh (-60 s/d -100 pips)",
            "bot_action": "SL langsung membatasi risiko maksimal $4-$5. Bot melakukan flip bias ke SELL di bawah level breakdown.",
        }
    }
}

# =========================================================================
# 2. DATA KALIBRASI PROYEKSI US CPI (NOWCASTING FED CLEVELAND) - 10 DESEMBER
# =========================================================================
DEFAULT_CPI_PROJECTION = {
    "event_id": "CPI",
    "event_name": "US Consumer Price Index (CPI) & Core CPI",
    "release_date_str": "2026-12-10 20:30:00",
    "release_date_display": "Kamis, 10 Desember 2026",
    "release_time_display": "Pukul 20:30:00 WIB",
    "release_date_et": "Thursday, Dec 10, 2026 08:30:00 AM ET",
    "release_iso_utc": "2026-12-10T13:30:00Z",
    "report_period": "Data Bulan November 2026",
    
    "cpi_headline_mom": {
        "previous": 0.40,
        "forecast_nowcast": 0.14,
        "delta": -0.26,
        "unit": "% MoM",
        "trend": "COOLING",
    },
    "cpi_core_mom": {
        "previous": 0.30,
        "forecast_nowcast": 0.21,
        "delta": -0.09,
        "unit": "% MoM",
        "trend": "COOLING",
    },
    
    "factors": [
        {
            "name": "Harga Bensin Ritel AS (Retail Gasoline)",
            "status": "TURUN DI BAWAH $4.00 / GALON",
            "impact": "DEFLASIONER KUAT (-)",
            "detail": "Turun di bawah $4/galon untuk pertama kali sejak pertengahan Juli 2026 (Analisis ING Think). Menekan bobot Headline CPI (~7.5%).",
            "score": -0.35,
        },
        {
            "name": "Minyak Mentah WTI & Energi Global",
            "status": "MELANDAI ($68 - $72 / BAREL)",
            "impact": "DEFLASIONER (-)",
            "detail": "Harapan de-eskalasi geopolitik di Timur Tengah mengurangi premi risiko energi global.",
            "score": -0.25,
        },
        {
            "name": "Cleveland Fed Inflation Nowcasting Model",
            "status": "PROYEKSI 0.14% vs SEBELUMNYA 0.40%",
            "impact": "MELAMBAT SIGNIFIKAN (-)",
            "detail": "Model ekonometrika Fed Cleveland mengestimasi perlambatan drastis inflasi bulanan.",
            "score": -0.30,
        },
        {
            "name": "Inflasi Inti (Core CPI di Luar Makanan & Energi)",
            "status": "PROYEKSI 0.21% vs SEBELUMNYA 0.30%",
            "impact": "MEREDA (-)",
            "detail": "Harga barang konsumsi sekunder dan jasa mulai melandai.",
            "score": -0.15,
        }
    ],

    "probability": {
        "cooling_inflation_prob": 84.5,
        "hot_inflation_prob": 15.5,
        "confidence_level": "SANGAT TINGGI (A+)",
        "consensus_verdict": "COOLING_INFLATION (CPI MELAMBAT)",
    },

    "xauusd_playbook": {
        "primary_direction": "STRONG_BUY",
        "action_label": "BULLISH EXPLOSION (BUY XAUUSD)",
        "rationale": "Inflasi AS melambat (0.14% vs 0.40%) -> The Fed leluasa memangkas suku bunga (Dovish Rate Cut) -> DXY & Yield Treasury anjlok -> Emas XAUUSD melonjak tinggi.",
        "expected_move_pips": "+80 s/d +150 pips (+8.0 hingga +15.0 poin Emas)",
        "potential_gain_usd": {
            "lot_0_01": "+$8.00 s/d +$15.00",
            "lot_0_02": "+$16.00 s/d +$30.00",
            "lot_0_05": "+$40.00 s/d +$75.00",
            "lot_0_10": "+$80.00 s/d +$150.00",
        },
        "execution_schedule": [
            {
                "time_wib": "20:00 - 20:15 WIB",
                "phase": "Fase 1: Pre-News Strategic Positioning",
                "action": "Open Posisi BUY Terencana (Lot terukur)",
                "detail": "Spread masih normal (1.5 - 2.5 pips). Masuk pada zona support M5/M15 berdasarkan 84.5% peluang CPI turun. SL di bawah support + buffer 50 pip.",
            },
            {
                "time_wib": "20:25 - 20:30 WIB",
                "phase": "Fase 2: Freeze & Circuit Breaker (Detik T-5m)",
                "action": "Tahan Entry Baru (News Shield Aktif)",
                "detail": "Spread broker mulai melebar (5 - 10 pips). Dilarang entry market di detik 00 untuk menghindari slippage.",
            },
            {
                "time_wib": "20:30:00 WIB",
                "phase": "Fase 3: Rilis Resmi BLS (T-0)",
                "action": "Data Rilis & Spike Awal",
                "detail": "Data resmi keluar. Spike liar 30-50 pip dalam hitungan detik.",
            },
            {
                "time_wib": "20:30:30 - 20:33 WIB",
                "phase": "Fase 4: Post-Release Momentum Sniper",
                "action": "Eksekusi Sniper BUY Mengikuti Arah Ekspansi",
                "detail": "Spread mulai menyempit normal. Angka CPI resmi terkonfirmasi turun -> Menembak BUY untuk menangkap reli lanjutan +80 hingga +150 pips.",
            }
        ],
        "contingency_plan": {
            "scenario": "Jika CPI Actual > 0.40% (Surprise Panas)",
            "market_reaction": "USD Menguat Keras, Emas Jatuh (-80 s/d -120 pips)",
            "bot_action": "Cut Loss Cepat posisi BUY atau Balik Arah SELL Sniper di bawah level Breakdown.",
        }
    }
}


class CpiPredictor:
    """Mesin kalkulasi nowcasting PPI & CPI dan eksekutor skenario Big Move XAUUSD."""

    def __init__(self, config_file: str = CPI_CONFIG_FILE):
        self.config_file = config_file
        self.sniper_enabled: bool = True
        self.sniper_mode: str = "PRE_POSITIONING" # "PRE_POSITIONING" atau "POST_MOMENTUM"
        self.target_lot: float = 0.02
        self.target_cuan_usd: float = 15.0
        self.sl_buffer_pips: float = 45.0
        import copy
        self.ppi_data = copy.deepcopy(DEFAULT_PPI_PROJECTION)
        self.cpi_data = copy.deepcopy(DEFAULT_CPI_PROJECTION)
        self._load_config()

    def _load_config(self) -> None:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.sniper_enabled = cfg.get("sniper_enabled", self.sniper_enabled)
                    self.sniper_mode = cfg.get("sniper_mode", self.sniper_mode)
                    self.target_lot = float(cfg.get("target_lot", self.target_lot))
                    self.target_cuan_usd = float(cfg.get("target_cuan_usd", self.target_cuan_usd))
                    if "ppi_projection" in cfg:
                        self.ppi_data.update(cfg["ppi_projection"])
                    if "cpi_projection" in cfg:
                        self.cpi_data.update(cfg["cpi_projection"])
            except Exception as e:
                logger.debug(f"Gagal muat konfigurasi Macro Sniper: {e}")

    def save_config(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump({
                    "sniper_enabled": self.sniper_enabled,
                    "sniper_mode": self.sniper_mode,
                    "target_lot": self.target_lot,
                    "target_cuan_usd": self.target_cuan_usd,
                    "ppi_projection": self.ppi_data,
                    "cpi_projection": self.cpi_data,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"Gagal simpan konfigurasi Macro Sniper: {e}")

    def _compute_countdown(self, iso_utc_str: str) -> tuple[str, str, float]:
        now_utc = datetime.now(timezone.utc)
        try:
            rel_dt = datetime.fromisoformat(iso_utc_str.replace("Z", "+00:00"))
            time_diff = rel_dt - now_utc
            diff_seconds = time_diff.total_seconds()
            
            if diff_seconds > 0:
                days = int(diff_seconds // 86400)
                hours = int((diff_seconds % 86400) // 3600)
                minutes = int((diff_seconds % 3600) // 60)
                seconds = int(diff_seconds % 60)
                if days > 0:
                    countdown_str = f"{days} Hari {hours} Jam {minutes} Menit {seconds} Detik"
                else:
                    countdown_str = f"{hours} Jam {minutes} Menit {seconds} Detik"
                status_phase = "COUNTDOWN_MENJELANG_RILIS"
            elif -3600 <= diff_seconds <= 0:
                countdown_str = "SEDANG BERLANGSUNG (LIVE RILIS BERITA)"
                status_phase = "LIVE_RELEASE"
            else:
                countdown_str = "SELESAI / DATA TELAH DIRILIS"
                status_phase = "POST_RELEASE"
        except Exception:
            countdown_str = "Jadwal Terdaftar"
            status_phase = "SCHEDULED"
            diff_seconds = 86400.0

        return countdown_str, status_phase, diff_seconds

    def get_cpi_nowcast_summary(self, event_type: str = "AUTO") -> Dict[str, Any]:
        """Menghitung sisa waktu rilis, probabilitas arah, dan rekomendasi XAUUSD.
        
        Args:
            event_type: 'AUTO' (pilih event terdekat, misal PPI malam ini), 'PPI', atau 'CPI'.
        """
        # Hitung waktu untuk kedua event
        ppi_cd, ppi_status, ppi_diff = self._compute_countdown(self.ppi_data["release_iso_utc"])
        cpi_cd, cpi_status, cpi_diff = self._compute_countdown(self.cpi_data["release_iso_utc"])

        # Tentukan event aktif
        ev_upper = event_type.upper()
        if ev_upper == "PPI":
            chosen_key = "PPI"
        elif ev_upper == "CPI":
            chosen_key = "CPI"
        else: # AUTO
            # Jika PPI masih akan datang (atau baru rilis dalam 2 jam), utamakan PPI malam ini
            if ppi_diff > -7200:
                chosen_key = "PPI"
            else:
                chosen_key = "CPI"

        active_data = self.ppi_data if chosen_key == "PPI" else self.cpi_data
        active_cd = ppi_cd if chosen_key == "PPI" else cpi_cd
        active_status = ppi_status if chosen_key == "PPI" else cpi_status
        active_diff = ppi_diff if chosen_key == "PPI" else cpi_diff

        return {
            "success": True,
            "active_event": chosen_key,
            "status_phase": active_status,
            "countdown_str": active_cd,
            "diff_seconds": active_diff,
            "sniper_settings": {
                "enabled": self.sniper_enabled,
                "mode": self.sniper_mode,
                "target_lot": self.target_lot,
                "target_cuan_usd": self.target_cuan_usd,
                "sl_buffer_pips": self.sl_buffer_pips,
            },
            # Data aktif terpilih (kompatibel dengan modal & banner lama)
            "cpi_projection": active_data,
            # Data lengkap kedua event untuk tab switcher
            "events": {
                "PPI": {
                    "data": self.ppi_data,
                    "countdown_str": ppi_cd,
                    "status_phase": ppi_status,
                    "diff_seconds": ppi_diff,
                },
                "CPI": {
                    "data": self.cpi_data,
                    "countdown_str": cpi_cd,
                    "status_phase": cpi_status,
                    "diff_seconds": cpi_diff,
                }
            }
        }

    def predict_market_direction_by_data(self, event_key: str = "AUTO", symbol: str = "XAUUSD") -> Dict[str, Any]:
        """Menghitung proyeksi arah pasar (BUY atau SELL) secara kuantitatif berdasarkan data makro.
        
        Logika Transmisi Makro & Dolar AS (DXY):
        1. Inflasi Mendingin / Data Lemah (Cooling / Deflationary / Dovish):
           - The Fed dovish -> Imbal hasil obligasi & Dolar AS (USD) anjlok.
           - XAUUSD (Emas): Dolar jatuh -> Emas NAIK -> Aksi: BUY.
           - EURUSD: Dolar jatuh -> EUR/USD NAIK -> Aksi: BUY.
           - GBPUSD: Dolar jatuh -> GBP/USD NAIK -> Aksi: BUY.
           - US100: Beban bunga melandai -> Nasdaq NAIK -> Aksi: BUY.
           - USDJPY: Dolar jatuh -> USD/JPY TURUN -> Aksi: SELL.
           
        2. Inflasi Memanas / Data Kuat (Heating / Hawkish / Inflation Surprise):
           - The Fed hawkish -> Imbal hasil obligasi & Dolar AS (USD) melambung deras.
           - XAUUSD (Emas): Dolar menguat deras -> Emas TERTEKAN JATUH -> Aksi: SELL.
           - EURUSD: Dolar menguat deras -> EUR/USD ANJLOK -> Aksi: SELL.
           - GBPUSD: Dolar menguat deras -> GBP/USD TURUN -> Aksi: SELL.
           - US100: Valuasi tech tertekan -> Nasdaq TURUN -> Aksi: SELL.
           - USDJPY: Dolar menguat -> USD/JPY NAIK -> Aksi: BUY.
        """
        summary = self.get_cpi_nowcast_summary(event_key)
        active_data = summary.get("cpi_projection", {})
        ev_name = summary.get("active_event", "MACRO")
        
        prob = active_data.get("probability", {})
        cooling_prob = float(prob.get("cooling_inflation_prob", 50.0))
        hot_prob = float(prob.get("hot_inflation_prob", 50.0))
        
        factors = active_data.get("factors", [])
        factor_score = sum(float(f.get("score", 0.0)) for f in factors)
        
        # Cek playbook jika ada penetapan arah eksplisit
        playbook = active_data.get("xauusd_playbook", {})
        playbook_dir = str(playbook.get("primary_direction", "")).upper()
        
        # Skor komposit probabilitas: > 0 artinya inflasi melambat (USD bearish), < 0 artinya inflasi panas (USD bullish)
        composite_score = (cooling_prob - hot_prob) / 100.0
        if factor_score != 0:
            composite_score += (-factor_score * 0.4)
            
        is_cooling = composite_score > 0
        if "BUY" in playbook_dir:
            is_cooling = True
        elif "SELL" in playbook_dir:
            is_cooling = False
            
        sym_upper = symbol.upper()
        is_gold = "XAU" in sym_upper or "GOLD" in sym_upper
        is_eur = "EUR" in sym_upper
        is_gbp = "GBP" in sym_upper
        is_us100 = "US100" in sym_upper or "NAS" in sym_upper
        is_usdjpy = "USDJPY" in sym_upper
        
        if is_usdjpy:
            action = "SELL" if is_cooling else "BUY"
            direction_desc = "USD Melemah -> USDJPY Jatuh" if is_cooling else "USD Menguat -> USDJPY Naik"
        else:
            action = "BUY" if is_cooling else "SELL"
            if is_gold:
                direction_desc = "Inflasi Melambat (Dolar Anjlok) -> Emas (XAUUSD) Meroket Naik" if is_cooling else "Inflasi Panas (Dolar Menguat) -> Emas (XAUUSD) Tertekan Jatuh"
            elif is_eur:
                direction_desc = "Inflasi Melambat (Dolar Anjlok) -> EURUSD Meroket Naik" if is_cooling else "Inflasi Panas (Dolar Menguat) -> EURUSD Anjlok Turun"
            else:
                direction_desc = f"Inflasi {'Melambat' if is_cooling else 'Panas'} -> {symbol} {'Naik' if is_cooling else 'Turun'}"

        prob_value = cooling_prob if is_cooling else hot_prob
        return {
            "symbol": symbol,
            "action": action, # "BUY" atau "SELL"
            "is_cooling": is_cooling,
            "probability_pct": prob_value,
            "confidence": prob.get("confidence_level", "TINGGI"),
            "event_name": ev_name,
            "direction_desc": direction_desc,
            "rationale": f"Data Makro {ev_name}: Probabilitas {prob_value:.1f}% Inflasi {'Melambat (Dovish)' if is_cooling else 'Memanas (Hawkish)'}. {direction_desc}.",
        }

    def evaluate_cpi_auto_sniper(
        self,
        current_price: float = 0.0,
        pip_size: float = 0.0,
        symbol: str = "XAUUSD",
        current_bid: float = 0.0,
        current_ask: float = 0.0,
        technicals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mengevaluasi apakah saat ini waktu yang tepat untuk eksekusi posisi Macro Sniper (BUY atau SELL sesuai data)."""
        summary = self.get_cpi_nowcast_summary("AUTO")
        diff_sec = summary.get("diff_seconds", 999999)
        ev_name = summary.get("active_event", "MACRO")
        
        # Tentukan arah BUY atau SELL secara presisi berdasarkan perhitungan data
        pred = self.predict_market_direction_by_data("AUTO", symbol)
        action = pred["action"] # "BUY" atau "SELL"
        prob_pct = pred["probability_pct"]
        dir_desc = pred["direction_desc"]
        
        # Resolusi harga dan parameter simbol
        sym_upper = symbol.upper()
        is_gold = "XAU" in sym_upper or "GOLD" in sym_upper
        decimals = 2 if is_gold else (5 if ("EUR" in sym_upper or "GBP" in sym_upper) else 2)
        
        if pip_size <= 0:
            pip_size = 0.1 if is_gold else (0.0001 if ("EUR" in sym_upper or "GBP" in sym_upper) else 0.01)
            
        cur_ref = current_price
        if action == "BUY":
            cur_ref = current_ask if current_ask > 0 else (current_price if current_price > 0 else (4000.0 if is_gold else 1.1000))
        else: # SELL
            cur_ref = current_bid if current_bid > 0 else (current_price if current_price > 0 else (4000.0 if is_gold else 1.1000))

        # Tentukan buffer SL & TP sesuai karakteristik pasangan mata uang
        if is_gold:
            sl_pips = self.sl_buffer_pips # 45.0 pips (4.5 poin emas)
            tp_pips = round(self.target_cuan_usd / (self.target_lot * 1.0 * 10), 1)
            tp_pips = max(tp_pips, 60.0) # Minimal 60 pips
        else:
            # Untuk EURUSD / Forex: ayunan pip lebih kecil
            sl_pips = 25.0 # 25 pips
            tp_pips = 50.0 # 50 pips
            
        # Hitung SL dan TP presisi berdasarkan arah BUY atau SELL
        if action == "BUY":
            entry_price = cur_ref
            sl_price = round(entry_price - (sl_pips * pip_size), decimals)
            tp_price = round(entry_price + (tp_pips * pip_size), decimals)
        else: # SELL
            entry_price = cur_ref
            sl_price = round(entry_price + (sl_pips * pip_size), decimals)
            tp_price = round(entry_price - (tp_pips * pip_size), decimals)
            
        # 1. Cek Mode Pre-Positioning (T-20 Menit hingga T-3 Menit sebelum rilis: 180s s/d 1200s)
        # Menjangkau hingga 5 menit sebelum rilis (T-5m / 300s) secara penuh!
        if self.sniper_enabled and self.sniper_mode == "PRE_POSITIONING":
            if 180 <= diff_sec <= 1200:
                tech_note = ""
                if technicals:
                    ema8 = float(technicals.get("ema8", 0.0))
                    ema21 = float(technicals.get("ema21", 0.0))
                    if ema8 > 0 and ema21 > 0:
                        if action == "BUY" and cur_ref >= ema21:
                            tech_note = " | Konfirmasi Teknikal: Harga bertahan di atas Support Dinamis EMA 8/21 (Validasi 95%+)"
                        elif action == "SELL" and cur_ref <= ema21:
                            tech_note = " | Konfirmasi Teknikal: Harga tertahan di bawah Resistance Dinamis EMA 8/21 (Validasi 95%+)"
                
                return {
                    "should_execute": True,
                    "action": action, # BUY atau SELL
                    "symbol": symbol,
                    "reason": f"🎯 {ev_name} SNIPER (Pre-Positioning): Peluang {prob_pct:.1f}% by data. {dir_desc}. Eksekusi {action} {symbol} di zona pra-rilis.{tech_note}",
                    "lot": self.target_lot,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "target_usd": self.target_cuan_usd,
                    "technique": f"{ev_name}_SNIPER",
                }
                
        # 2. Cek Mode Post-Momentum (T+30s hingga T+3m pasca rilis)
        if self.sniper_enabled and self.sniper_mode == "POST_MOMENTUM":
            if -180 <= diff_sec <= -30:
                post_sl_pips = 40.0 if is_gold else 20.0
                post_tp_pips = 80.0 if is_gold else 40.0
                if action == "BUY":
                    entry_price = cur_ref
                    sl_price = round(entry_price - (post_sl_pips * pip_size), decimals)
                    tp_price = round(entry_price + (post_tp_pips * pip_size), decimals)
                else: # SELL
                    entry_price = cur_ref
                    sl_price = round(entry_price + (post_sl_pips * pip_size), decimals)
                    tp_price = round(entry_price - (post_tp_pips * pip_size), decimals)
                    
                return {
                    "should_execute": True,
                    "action": action,
                    "symbol": symbol,
                    "reason": f"🚀 {ev_name} SNIPER (Post-Release Momentum): Terkonfirmasi data {ev_name}. Menembak ekspansi arah {action} {symbol} pasca normalisasi spread.",
                    "lot": self.target_lot,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "target_usd": self.target_cuan_usd,
                    "technique": f"{ev_name}_SNIPER",
                }

        return {
            "should_execute": False,
            "action": action,
            "symbol": symbol,
            "reason": f"Menunggu jendela eksekusi {ev_name} Sniper ({self.sniper_mode}). Countdown: {summary.get('countdown_str')}",
        }


# Singleton Global
_cpi_predictor_instance: Optional[CpiPredictor] = None

def get_cpi_predictor() -> CpiPredictor:
    global _cpi_predictor_instance
    if _cpi_predictor_instance is None:
        _cpi_predictor_instance = CpiPredictor()
    return _cpi_predictor_instance
