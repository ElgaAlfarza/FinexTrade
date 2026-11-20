"""Live Trading Engine: Otomasi siklus trading real-time (Tahap 3).

Menjalankan trading loop:
1. Monitoring bar baru (candle close detection).
2. Evaluasi sinyal Fibonacci + Rejection.
3. Eksekusi order via MT5 atau Paper Simulator.
4. Auto Break-Even & Trailing Stop tracker setiap 1 detik.
5. Circuit breaker harian & emergency kill-switch.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import csv
import numpy as np
import pandas as pd

from config import AVAILABLE_TECHNIQUES, PRESETS, SYMBOL_SPECS, StrategyParams, build_params, symbol_decimals
from execution import (
    IExecutor,
    MT5Executor,
    OrderManager,
    PaperExecutor,
    Position,
    estimate_pip_value,
    get_symbol_pip_size,
    infer_technique_from_text,
    extract_techniques_from_trade,
    get_session_reset_info,
    TRADE_RECORDS_CSV,
    TRADE_RECORDS_ARCHIVE_CSV,
)
from mt5_client import MT5Session, load_env
import dataclasses
from strategy import align_htf, generate_signals, htf_structure, ma_direction, ma_htf_frame, resample_ohlc
from learner import Learner, THRESHOLD as THRESHOLD_GATE

logger = logging.getLogger("finex.engine")


def get_current_market_session(utc_dt: Optional[datetime] = None) -> dict:
    """Deteksi Sesi Pasar Global (Waktu UTC):
    - Asian Session: 00:00 - 08:00 UTC (07:00 - 15:00 WIB) -> Likuiditas Rendah / Ranging -> Status: WAIT
    - London Session: 08:00 - 12:00 UTC (15:00 - 19:00 WIB) -> Likuiditas Meningkat & Tren Terbentuk -> Status: ACTIVE
    - Overlap London & New York: 12:00 - 17:00 UTC (19:00 - 00:00 WIB) -> Prime Trading Window (Volatilitas & Likuiditas Tertinggi) -> Status: VERY_ACTIVE
    - Late New York: 17:00 - 21:00 UTC (00:00 - 04:00 WIB) -> Pergerakan Melemah -> Status: STRICT_ACTIVE
    - Sydney / Pasca-NY: 21:00 - 24:00 UTC (04:00 - 07:00 WIB) -> Rollover / Off-market -> Status: WAIT
    """
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)
    t_float = utc_dt.hour + utc_dt.minute / 60.0

    if 0.0 <= t_float < 8.0:
        return {
            "session": "Asian",
            "name": "Sesi Asia (Tokyo/Sydney)",
            "status": "WAIT",
            "is_active_session": False,
            "liquidity": "Rendah / Ranging",
            "icon": "🌏",
            "prime_window": False,
            "hours_utc": "00:00 - 08:00 UTC",
            "hours_wib": "07:00 - 15:00 WIB",
            "detail": "Likuiditas rendah & ranging. Menunggu Sesi London (08:00 UTC / 15:00 WIB).",
        }
    elif 8.0 <= t_float < 12.0:
        return {
            "session": "London",
            "name": "Sesi London (Eropa)",
            "status": "ACTIVE",
            "is_active_session": True,
            "liquidity": "Tinggi / Tren Terbentuk",
            "icon": "🏛️",
            "prime_window": False,
            "hours_utc": "08:00 - 12:00 UTC",
            "hours_wib": "15:00 - 19:00 WIB",
            "detail": "Arah tren mulai terbentuk dengan likuiditas tinggi.",
        }
    elif 12.0 <= t_float < 17.0:
        return {
            "session": "London & New York Overlap",
            "name": "Overlap London & New York",
            "status": "VERY_ACTIVE",
            "is_active_session": True,
            "liquidity": "Tertinggi (Prime Window)",
            "icon": "🔥",
            "prime_window": True,
            "hours_utc": "12:00 - 17:00 UTC",
            "hours_wib": "19:00 - 00:00 WIB",
            "detail": "Prime Window: Volatilitas & likuiditas emas & forex tertinggi.",
        }
    elif 17.0 <= t_float < 21.0:
        return {
            "session": "Late New York",
            "name": "Sesi New York Akhir",
            "status": "STRICT_ACTIVE",
            "is_active_session": True,
            "liquidity": "Sedang / Melemah",
            "icon": "🗽",
            "prime_window": False,
            "hours_utc": "17:00 - 21:00 UTC",
            "hours_wib": "00:00 - 04:00 WIB",
            "detail": "Pergerakan pasar stabil menuju penutupan US.",
        }
    else:
        return {
            "session": "Rollover / Pasca-NY",
            "name": "Sesi Pasca-NY / Rollover",
            "status": "WAIT",
            "is_active_session": False,
            "liquidity": "Sangat Rendah (Spread Melebar)",
            "icon": "🌙",
            "prime_window": False,
            "hours_utc": "21:00 - 24:00 UTC",
            "hours_wib": "04:00 - 07:00 WIB",
            "detail": "Spread broker melebar menjelang penutupan harian.",
        }


class LiveBotEngine:
    """Mesin utama trading otomatis."""

    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M5",
        use_paper: bool = False,
        params: Optional[StrategyParams] = None,
    ):
        self.symbol = (symbol or "XAUUSD").upper()
        self.timeframe = (timeframe or "M5").upper()
        self.use_paper = use_paper

        # Pengaturan yang harus bertahan saat simbol/timeframe diganti (mode latihan)
        self.param_overrides: dict = {}
        self.bars_since_explore: int = 10**6
        self.pending_explore: set = set()

        self.params = params or self._build_params()
        self.executor: IExecutor = self._init_executor()
        db_path = os.environ.get("FINEX_BOT_STATE_DB")
        if not db_path:
            db_path = ":memory:" if self.use_paper else None
        self.order_manager = OrderManager(self.executor, self.params, db_path=db_path)

        self.is_running: bool = False
        self.is_paused: bool = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # State data candle
        self.df_candles: pd.DataFrame = pd.DataFrame()
        self.last_candle_time: Optional[pd.Timestamp] = None
        self.symbol_supported: bool = True
        self.current_bid: float = float(self.params.default_price)
        self.current_ask: float = float(self.params.default_price + self.params.spread_pips * self.params.pip_size)

        # Telemetry callbacks untuk WebSocket dashboard
        self.telemetry_subscribers: List[Callable[[dict], None]] = []
        self.learner = Learner()
        self.learner_explore = Learner("learner_state_explore.json")  # model terpisah untuk entry mode latihan
        self.pending_features: Dict[int, dict] = {}
        self.last_diag: str = "Menunggu candle pertama..."
        self.last_scalp_ts: float = 0.0
        self.active_signal: dict = {}

        # Fokus Eksklusif Tunggal: XAUUSD (Emas) M1 Saja
        self.watchlist: List[str] = ["XAUUSD"]
        self.multi_symbol_enabled: bool = False
        self.max_total_open_positions: int = int(getattr(self.params, "scalping_max_open", 5))
        self.max_open_per_symbol: int = int(getattr(self.params, "scalping_max_per_symbol", 5))
        self.last_loss_by_symbol: Dict[str, dict] = {}
        self.last_closed_trade_ts_by_symbol: Dict[str, float] = {}
        self.last_scalp_by_symbol: Dict[str, float] = {}

    def _init_executor(self) -> IExecutor:
        load_env()
        if not self.use_paper:
            try:
                login_str = os.environ.get("MT5_LOGIN", "")
                if login_str and login_str.isdigit():
                    login = int(login_str)
                    pwd = os.environ.get("MT5_PASSWORD", "")
                    srv = os.environ.get("MT5_SERVER", "FinexBisnisSolusi-Demo")
                    acc_type = os.environ.get("MT5_ACCOUNT_TYPE", "demo").lower()
                    is_demo = (acc_type == "demo")
                    term_path = os.environ.get("MT5_PATH") or None
                    session = MT5Session(login, pwd, srv, terminal_path=term_path, require_demo=is_demo)
                    session.__enter__()
                    logger.info(f"Terhubung ke terminal MetaTrader 5 ({acc_type.upper()})!")
                    return MT5Executor(session)
            except Exception as e:
                logger.error(f"Inisialisasi MT5 gagal ({e}). Beralih ke Paper Trading Virtual Simulator (koneksi broker tidak aktif).")
                self.use_paper = True

        logger.info("Menggunakan Paper Trading Simulator.")
        return PaperExecutor(initial_balance=10000.0)

    def switch_to_mt5(self, login: int, password: str, server: str, account_type: str = "demo", terminal_path: str | None = None) -> dict:
        """Hubungkan ke akun MT5 (Demo atau Real) langsung dari runtime."""
        is_demo = (account_type.lower() == "demo")
        allow_real = (os.environ.get("ALLOW_LIVE_REAL", "0") == "1")
        if not is_demo and not allow_real:
            raise PermissionError("Beralih ke akun REAL diblokir demi keamanan modal. Butuh ALLOW_LIVE_REAL=1.")
        term_path = terminal_path.strip() if terminal_path and terminal_path.strip() else None

        # Smart Validation: Jika pengguna memasukkan host:port (seperti prod-mt5-demo1.fnx.xmt.mx:443) ke kolom terminal_path
        if term_path and (":443" in term_path or "fnx." in term_path or not term_path.lower().endswith(".exe")):
            # Anggap itu alamat host server alternatif
            if not server or server == "FinexBisnisSolusi-Demo":
                server = term_path
            term_path = None

        # Auto-detect path terminal jika tidak diisi
        if not term_path:
            default_paths = [
                os.environ.get("MT5_PATH"),
                r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe",
                r"C:\Program Files\MetaTrader 5\terminal64.exe",
            ]
            for p in default_paths:
                if p and os.path.exists(p):
                    term_path = p
                    break

        # Coba login langsung ke akun target via sesi MT5 aktif (tanpa shutdown IPC)
        login_success = False
        if isinstance(self.executor, MT5Executor):
            try:
                mt5_mod = self.executor.mt5
                if mt5_mod.terminal_info() is not None:
                    # MT5 terminal masih aktif, login langsung via IPC yang ada
                    if mt5_mod.login(login=login, password=password, server=server):
                        info = mt5_mod.account_info()
                        if info is not None:
                            if is_demo and info.trade_mode != mt5_mod.ACCOUNT_TRADE_MODE_DEMO:
                                raise RuntimeError("Akun ini BUKAN akun demo. Koneksi dihentikan demi keamanan.")
                            login_success = True
                            self.executor.s.login = login
                            self.executor.s.password = password
                            self.executor.s.server = server
                            self.executor.s.require_demo = is_demo
            except Exception as e:
                logger.warning(f"Login langsung via terminal aktif gagal ({e}), mencoba re-inisialisasi...")
                login_success = False

        if not login_success:
            # Jika belum terkoneksi atau login langsung gagal, tutup sesi lama dan inisialisasi sesi baru
            if isinstance(self.executor, MT5Executor):
                try:
                    self.executor.s.__exit__(None, None, None)
                except Exception:
                    pass
                time.sleep(0.5)

            session = MT5Session(login, password, server, terminal_path=term_path, require_demo=is_demo)
            session.__enter__()
            new_executor = MT5Executor(session)
            self.executor = new_executor
            self.order_manager.executor = new_executor

        self.use_paper = False

        # Langsung muat candle live dari MT5 dan update harga
        self.load_initial_candles()
        try:
            with self.executor._lock:
                tick = self.executor.mt5.symbol_info_tick(self.symbol)
            if tick:
                self.current_bid = tick.bid
                self.current_ask = tick.ask
                self.executor.update_prices(self.symbol, tick.bid, tick.ask)
        except Exception:
            pass

        # Simpan kredensial ke file .env agar tersimpan permanen
        self._save_env(login, password, server, account_type, term_path)
        self._ensure_mt5_algo_config()

        acc = self.executor.get_account_info()
        self.order_manager.log(f"LOGIN SUKSES: MT5 {account_type.upper()} #{login} ({server}) | Saldo: ${acc.get('balance', 0):,.2f}", "INFO")
        return acc

    def switch_to_paper(self) -> dict:
        """Putuskan MT5 dan beralih ke Paper Trading Simulator."""
        if isinstance(self.executor, MT5Executor):
            try:
                self.executor.s.__exit__(None, None, None)
            except Exception:
                pass
        self.executor = PaperExecutor(initial_balance=10000.0)
        self.order_manager.executor = self.executor
        self.use_paper = True
        self.order_manager.log("Beralih kembali ke mode Paper Trading Simulator (Akun Virtual)", "INFO")
        return self.executor.get_account_info()

    def _save_env(self, login: int, password: str, server: str, account_type: str, terminal_path: str | None = None) -> None:
        """Simpan konfigurasi sesi aktif ke environment memori saja (TANPA menulis password ke disk)."""
        os.environ["MT5_LOGIN"] = str(login)
        os.environ["MT5_SERVER"] = str(server)
        os.environ["MT5_ACCOUNT_TYPE"] = str(account_type).lower()
        if terminal_path:
            os.environ["MT5_PATH"] = str(terminal_path)

    def _ensure_mt5_algo_config(self) -> None:
        """Kunci konfigurasi MT5 agar Algo Trading selalu aktif permanen dan tidak mati saat ganti akun."""
        try:
            import glob
            appdata = os.environ.get("APPDATA", "")
            ini_files = glob.glob(os.path.join(appdata, "MetaQuotes", "Terminal", "*", "config", "common.ini"))
            for p in ini_files:
                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-16") as f:
                            c = f.read()
                        if "Account=1" in c or "Profile=1" in c or "Chart=1" in c or "Enabled=0" in c or "Api=1" in c:
                            c = c.replace("Account=1", "Account=0")
                            c = c.replace("Profile=1", "Profile=0")
                            c = c.replace("Chart=1", "Chart=0")
                            c = c.replace("Api=1", "Api=0")
                            c = c.replace("Enabled=0", "Enabled=1")
                            with open(p, "w", encoding="utf-16") as f:
                                f.write(c)
                            logger.info(f"Otomatis mengunci Algo Trading & External API AKTIF di MT5 ({p})")
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Gagal periksa config MT5 algo: {e}")

    def subscribe_telemetry(self, cb: Callable[[dict], None]) -> None:
        self.telemetry_subscribers.append(cb)

    def broadcast(self, event_type: str, data: Any) -> None:
        payload = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        for sub in list(self.telemetry_subscribers):
            try:
                sub(payload)
            except Exception:
                pass

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.is_paused = False
        self._ensure_mt5_algo_config()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="FinexBotLoop")
        self._thread.start()
        self.order_manager.log(f"Bot dimulai pada {self.symbol} ({self.timeframe}) | Mode: {self.get_mode()}", "INFO")
        try:
            if isinstance(self.executor, MT5Executor):
                mt5_mod = getattr(self.executor, "mt5", None)
                if mt5_mod:
                    t_info = mt5_mod.terminal_info()
                    if t_info and t_info.trade_allowed:
                        self.order_manager.log("🤖 MT5 Algo Trading: AKTIF OTOMATIS & TERKUNCI (AutoTrading Enabled)", "INFO")
                    elif t_info:
                        self.order_manager.log("⚠️ MT5 Algo Trading Belum Aktif di Toolbar MT5! Tekan tombol 'Algo Trading' (Ctrl+E) di MT5.", "WARNING")
        except Exception:
            pass

    def pause(self) -> None:
        self.is_paused = True
        self.order_manager.log("Bot di-pause (tidak akan membuka trade baru).", "WARNING")

    def resume(self) -> None:
        self.is_paused = False
        self._ensure_mt5_algo_config()
        self.order_manager.log("Bot di-resume.", "INFO")
        try:
            if isinstance(self.executor, MT5Executor):
                mt5_mod = getattr(self.executor, "mt5", None)
                if mt5_mod:
                    t_info = mt5_mod.terminal_info()
                    if t_info and t_info.trade_allowed:
                        self.order_manager.log("🤖 MT5 Algo Trading: AKTIF OTOMATIS & TERKUNCI (AutoTrading Enabled)", "INFO")
                    elif t_info:
                        self.order_manager.log("⚠️ MT5 Algo Trading Belum Aktif di Toolbar MT5! Tekan tombol 'Algo Trading' (Ctrl+E) di MT5.", "WARNING")
        except Exception:
            pass

    def stop(self) -> None:
        self.is_running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.order_manager.log("Bot dihentikan.", "INFO")

    def _build_params(self) -> StrategyParams:
        return build_params(self.symbol, self.timeframe, self.param_overrides)

    def set_explore(self, enabled: bool, every_bars: Optional[int] = None,
                    max_open: Optional[int] = None, daily_loss_r: Optional[float] = None) -> None:
        """Aktif/nonaktifkan MODE LATIHAN. Hanya boleh di akun DEMO atau PAPER."""
        if enabled and self.get_mode() == "MT5_REAL":
            raise ValueError("Mode latihan hanya untuk akun DEMO/PAPER. Akun REAL terdeteksi, ditolak.")
        if every_bars is not None and int(every_bars) < 1:
            raise ValueError("every_bars minimal 1")
        if max_open is not None and int(max_open) < 1:
            raise ValueError("max_open minimal 1")
        if daily_loss_r is not None and float(daily_loss_r) <= 0:
            raise ValueError("daily_loss_r harus > 0")
        ov = self.param_overrides
        ov["explore_mode"] = bool(enabled)
        if every_bars is not None:
            ov["explore_every_bars"] = int(every_bars)
        if max_open is not None:
            ov["explore_max_open"] = int(max_open)
        if daily_loss_r is not None:
            ov["explore_daily_loss_r"] = float(daily_loss_r)
        self.params = dataclasses.replace(self.params, **ov)
        self.order_manager.params = self.params
        p = self.params
        self.order_manager.log(
            f"MODE LATIHAN {'AKTIF' if p.explore_mode else 'NONAKTIF'} | tiap {p.explore_every_bars} candle {self.timeframe} | "
            f"maks {p.explore_max_open} posisi | batas rugi harian -{p.explore_daily_loss_r}R", "RISK_ALERT" if p.explore_mode else "INFO")

    def get_explore_status(self) -> dict:
        p = self.params
        return {"enabled": bool(p.explore_mode), "every_bars": p.explore_every_bars, "max_open": p.explore_max_open,
                "daily_loss_r": p.explore_daily_loss_r, "timeframe": self.timeframe,
                "learner": self.learner_explore.stats()}

    def emergency_kill_switch(self) -> int:
        """Tutup semua posisi seketika dan pause bot."""
        self.pause()
        closed_count = self.executor.close_all(reason="EMERGENCY_KILL_SWITCH")
        self.order_manager.log(f"KILL SWITCH AKTIF: {closed_count} posisi ditutup paksa!", "RISK_ALERT")
        return closed_count

    def set_active_technique(self, tech: str = "EMA_TREND") -> dict:
        """Pilih teknik trading spesifik (fokus tunggal EMA_TREND)."""
        tech_id = "EMA_TREND"
        self.param_overrides["active_technique"] = tech_id
        self.params = dataclasses.replace(self.params, active_technique=tech_id)
        self.order_manager.params = self.params
        meta = AVAILABLE_TECHNIQUES.get(tech_id, {})
        self.order_manager.log(
            f"🎯 TEKNIK AKTIF DIPILIH: {meta.get('icon', '📈')} {meta.get('name', tech_id)}",
            "INFO",
        )
        self._compute_active_signal()
        return {
            "status": "success",
            "active_technique": tech_id,
            "meta": meta,
        }

    def set_session_filter_mode(self, mode: str) -> dict:
        """Atur mode filter sesi pasar: 'STRICT' (hanya London/NY) atau 'ALL' (bebas jam trading)."""
        clean_mode = str(mode or "STRICT").upper().strip()
        if clean_mode not in ("STRICT", "ALL"):
            clean_mode = "STRICT"
        self.param_overrides["session_filter_mode"] = clean_mode
        self.param_overrides["use_session_filter"] = (clean_mode == "STRICT")
        self.params = dataclasses.replace(self.params, session_filter_mode=clean_mode, use_session_filter=(clean_mode == "STRICT"))
        self.order_manager.params = self.params
        session_info = get_current_market_session()
        self.order_manager.log(
            f"🌐 FILTER SESI PASAR: {clean_mode} (Saat ini: {session_info['name']} - {session_info['status']})",
            "INFO",
        )
        self._compute_active_signal()
        return {
            "status": "success",
            "session_filter_mode": clean_mode,
            "market_session": session_info,
        }

    def set_multi_symbol_mode(self, enabled: bool) -> dict:
        """Atur cakupan trading pasar: True = Scan semua market (multi-pair), False = Fokus 1 market layar."""
        self.multi_symbol_enabled = bool(enabled)
        self.param_overrides["scalping_multi_symbol"] = self.multi_symbol_enabled
        self.params = dataclasses.replace(self.params, scalping_multi_symbol=self.multi_symbol_enabled)
        self.order_manager.params = self.params
        mode_name = "SEMUA MARKET (Multi-Pair)" if self.multi_symbol_enabled else f"FOKUS 1 MARKET ({self.symbol})"
        self.order_manager.log(
            f"🌐 CAKUPAN MARKET: Mode diubah ke {mode_name}",
            "INFO",
        )
        return {
            "status": "success",
            "multi_symbol_enabled": self.multi_symbol_enabled,
            "mode": "ALL" if self.multi_symbol_enabled else "SINGLE",
            "active_symbol": self.symbol,
            "watchlist": getattr(self, "watchlist", ["XAUUSD", "EURUSD", "GBPUSD", "US100"]),
        }

    def set_max_positions_mode(self, mode: str) -> dict:
        """Atur kuota open posisi: '1', '2', '3', '4', '5', atau 'AUTO' (Smart Scale-In/Piramid saat Mendekati TP)."""
        clean_mode = str(mode).strip().upper()
        if clean_mode not in ("1", "2", "3", "4", "5", "AUTO"):
            clean_mode = "AUTO"

        if clean_mode in ("1", "2", "3", "4", "5"):
            n = int(clean_mode)
            self.max_total_open_positions = n
            self.max_open_per_symbol = n
            self.param_overrides["scalping_max_open"] = n
            self.param_overrides["scalping_max_per_symbol"] = n
            self.param_overrides["scalping_max_open_mode"] = clean_mode
            self.params = dataclasses.replace(
                self.params,
                scalping_max_open=n,
                scalping_max_per_symbol=n,
                scalping_max_open_mode=clean_mode,
            )
            mode_desc = f"MAKSIMAL {n} POSISI"
        else:
            self.max_total_open_positions = 5
            self.max_open_per_symbol = 5
            self.param_overrides["scalping_max_open"] = 5
            self.param_overrides["scalping_max_per_symbol"] = 5
            self.param_overrides["scalping_max_open_mode"] = "AUTO"
            self.params = dataclasses.replace(
                self.params,
                scalping_max_open=5,
                scalping_max_per_symbol=5,
                scalping_max_open_mode="AUTO",
            )
            mode_desc = "OTOMATIS (Smart Scale-In / Piramid TP)"

        self.order_manager.params = self.params
        self.order_manager.log(
            f"🎯 KUOTA POSISI: Mode diatur ke [{mode_desc}]",
            "INFO"
        )
        return {
            "status": "success",
            "mode": clean_mode,
            "max_total_open": self.max_total_open_positions,
            "description": mode_desc
        }

    def _check_auto_pyramid_allowed(self, all_positions: list) -> tuple[bool, str]:
        """
        Kaidah Mode Otomatis (Smart Pyramiding / Scale-In):
        Selagi ada posisi yang sedang berjalan dan berkemungkinan besar TP
        (cushion profit positif, mendekati TP >= 25%, atau Break-Even aktif),
        bot diperbolehkan membuka posisi tambahan hingga maksimal 5 posisi.
        Jika posisi yang ada masih minus atau belum aman, dilarang tambah beban.
        """
        if not all_positions:
            return True, "Tidak ada posisi aktif, entry baru diperbolehkan."

        if len(all_positions) >= 5:
            return False, "Kuota maksimal 5 posisi tercapai."

        # Cek apakah ada posisi yang sedang floating minus
        for p in all_positions:
            p_pnl = float(getattr(p, "floating_pnl", 0.0) or 0.0)
            if p_pnl < -0.20:
                ticket = getattr(p, "ticket", 0)
                return False, f"Posisi #{ticket} sedang floating minus (-${abs(p_pnl):.2f}). Menolak scale-in demi proteksi modal."

        # Evaluasi posisi terbaru
        latest_pos = all_positions[-1]
        pnl = float(getattr(latest_pos, "floating_pnl", 0.0) or 0.0)
        be_active = bool(getattr(latest_pos, "be_active", False))
        floating_r = float(getattr(latest_pos, "floating_r", 0.0) or 0.0)
        open_px = float(getattr(latest_pos, "open_price", 0.0) or 0.0)
        cur_px = float(getattr(latest_pos, "current_price", 0.0) or 0.0)
        tp_px = float(getattr(latest_pos, "tp", 0.0) or 0.0)
        side = getattr(latest_pos, "side", "BUY")

        progress_pct = 0.0
        if tp_px > 0 and open_px > 0:
            if side == "BUY" and tp_px > open_px:
                progress_pct = max(0.0, (cur_px - open_px) / (tp_px - open_px) * 100.0)
            elif side == "SELL" and tp_px < open_px:
                progress_pct = max(0.0, (open_px - cur_px) / (open_px - tp_px) * 100.0)

        is_heading_to_tp = be_active or (progress_pct >= 25.0) or (floating_r >= 0.35) or (pnl >= 1.50)

        if is_heading_to_tp:
            reason = f"Posisi #{getattr(latest_pos, 'ticket', 0)} menuju TP (Progress: {progress_pct:.0f}%, Cuan: +${pnl:.2f}) -> Scale-In diizinkan."
            return True, reason
        else:
            reason = f"Posisi #{getattr(latest_pos, 'ticket', 0)} belum cukup mendekati TP (Progress: {progress_pct:.0f}%, Cuan: ${pnl:.2f}). Menunggu konfirmasi cuan sebelum scale-in."
            return False, reason

    def get_technique_statistics(self) -> dict:
        """Kompilasi statistik performa & akurasi (Win Rate, Total Trade, TP vs SL, PnL) untuk setiap teknik."""
        active_tech = getattr(self.params, "active_technique", "ALL")
        stats = {}
        for k, meta in AVAILABLE_TECHNIQUES.items():
            stats[k] = {
                "id": k,
                "name": meta["name"],
                "short_name": meta["short_name"],
                "icon": meta["icon"],
                "desc": meta["desc"],
                "badge_color": meta["badge_color"],
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "pnl": 0.0,
                "is_active": (k == active_tech),
            }

        seen_tickets = set()
        today_date_str = datetime.now().strftime("%Y-%m-%d")

        # 1. Dari trade history executor hari ini agar 100% sinkron dengan performa harian
        try:
            hist = self.executor.get_trade_history()
            for t in hist:
                tk = str(t.get("ticket", ""))
                if tk and tk in seen_tickets:
                    continue
                c_time = str(t.get("close_time", "") or t.get("open_time", "") or "")
                if not c_time.startswith(today_date_str):
                    continue
                if tk:
                    seen_tickets.add(tk)
                pnl = float(t.get("pnl", 0.0) or t.get("profit", 0.0) or 0.0)
                matched_techs = extract_techniques_from_trade(t)
                for tech in matched_techs:
                    if tech in stats:
                        stats[tech]["total_trades"] += 1
                        if pnl > 0:
                            stats[tech]["wins"] += 1
                        elif pnl < 0:
                            stats[tech]["losses"] += 1
                        stats[tech]["pnl"] += pnl
        except Exception as e:
            logger.debug(f"Gagal baca trade history executor: {e}")

        # 2. Jika hari ini belum ada transaksi di executor, ambil dari catatan CSV hari ini
        if stats.get("EMA_TREND", {}).get("total_trades", 0) == 0 and os.path.exists(TRADE_RECORDS_CSV):
            try:
                with open(TRADE_RECORDS_CSV, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        tk = str(r.get("ticket", ""))
                        if tk and tk in seen_tickets:
                            continue
                        close_time = str(r.get("close_time", "") or r.get("open_time", "") or "")
                        if not close_time.startswith(today_date_str):
                            continue
                        if tk:
                            seen_tickets.add(tk)
                        try:
                            pnl = float(r.get("pnl", 0.0) or 0.0)
                        except ValueError:
                            pnl = 0.0
                        matched_techs = extract_techniques_from_trade(r)
                        for tech in matched_techs:
                            if tech in stats:
                                stats[tech]["total_trades"] += 1
                                if pnl > 0:
                                    stats[tech]["wins"] += 1
                                elif pnl < 0:
                                    stats[tech]["losses"] += 1
                                stats[tech]["pnl"] += pnl
            except Exception as e:
                logger.debug(f"Gagal baca file {TRADE_RECORDS_CSV}: {e}")

        tech_evals = self.evaluate_all_techniques()
        for k, v in stats.items():
            tot = v["total_trades"]
            v["win_rate"] = round((v["wins"] / tot * 100.0), 1) if tot > 0 else 0.0
            v["pnl"] = round(v["pnl"], 2)
            ev = tech_evals.get(k, {})
            v["action"] = ev.get("action", "WAIT")
            v["is_ready"] = ev.get("is_ready", False)
            v["signal_reason"] = ev.get("reason", "Menunggu konfirmasi setup...")
            v["confidence"] = ev.get("confidence", 50.0)
            v["entry_price"] = ev.get("entry_price")
            v["tp_price"] = ev.get("tp_price")
            v["sl_price"] = ev.get("sl_price")

        return stats

    def get_mode(self) -> str:
        acc = self.executor.get_account_info()
        return acc.get("mode", "PAPER")

    def set_market(self, symbol: str, timeframe: str) -> None:
        """Ganti instrumen simbol dan timeframe aktif."""
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.params = self._build_params()
        self.order_manager.params = self.params
        self.load_initial_candles()
        self.order_manager.log(f"🎯 FOKUS 1 MARKET: Bot 100% fokus hanya pada {self.symbol} ({self.timeframe})", "INFO")

    def _ensure_symbol(self) -> bool:
        if not isinstance(self.executor, MT5Executor):
            self.symbol_supported = True
            return True
        with self.executor._lock:
            mt5 = self.executor.mt5
            if mt5.symbol_select(self.symbol, True):
                self.symbol_supported = True
                return True
            # Auto-detect variasi simbol broker (contoh: XAUUSD_c, GOLD, XAUUSD.raw, XAUUSDm)
            all_syms = mt5.symbols_get() or []
            target_clean = self.symbol.upper().replace("/", "").replace("_", "")
            base_sym = target_clean
            for sfx in ["U", "C", "RAW", "M", "MICRO", "PRO"]:
                if base_sym.endswith(sfx) and len(base_sym) > len(sfx) + 3:
                    base_sym = base_sym[:-len(sfx)]
                    break

            for s in all_syms:
                s_name = s.name
                s_clean = s_name.upper().replace("/", "").replace("_", "")
                s_base = s_clean
                for sfx in ["U", "C", "RAW", "M", "MICRO", "PRO"]:
                    if s_base.endswith(sfx) and len(s_base) > len(sfx) + 3:
                        s_base = s_base[:-len(sfx)]
                        break

                if (s_clean == target_clean or 
                    s_base == base_sym or 
                    (base_sym and base_sym in s_clean) or 
                    (s_base and s_base in target_clean) or 
                    (target_clean.startswith("XAU") and "GOLD" in s_clean)):
                    if mt5.symbol_select(s_name, True):
                        logger.info(f"Otomatis memetakan simbol '{self.symbol}' ke broker symbol '{s_name}'")
                        self.symbol = s_name
                        self.symbol_supported = True
                        return True
            cands = [x.name for x in all_syms if x.name.upper().startswith(self.symbol.upper()[:4])][:5]
        self.symbol_supported = False
        self.order_manager.log(f"Simbol {self.symbol} tidak tersedia di broker MT5. Kandidat: {cands or 'Forex/Emas/US100'}", "WARNING")
        return False

    def load_initial_candles(self, csv_path: Optional[str] = None) -> None:
        """Muat data candle awal dari MT5 (jika terhubung & didukung), CSV simbol terkait, atau sintetis."""
        self.df_candles = pd.DataFrame()
        supported = self._ensure_symbol()
        if isinstance(self.executor, MT5Executor) and supported:
            try:
                with self.executor._lock:
                    mt5 = self.executor.mt5
                    tf_map = {
                        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                        "D1": mt5.TIMEFRAME_D1, "W1": getattr(mt5, "TIMEFRAME_W1", mt5.TIMEFRAME_D1),
                        "MN": getattr(mt5, "TIMEFRAME_MN1", mt5.TIMEFRAME_D1),
                    }
                    tf_val = tf_map.get(self.timeframe, mt5.TIMEFRAME_M5)
                    rates = mt5.copy_rates_from_pos(self.symbol, tf_val, 0, 3000)
                    tick = mt5.symbol_info_tick(self.symbol)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    self.df_candles = df.set_index("time").sort_index()
                    last_row = self.df_candles.iloc[-1]
                    self.last_candle_time = self.df_candles.index[-1]
                    if tick and tick.bid > 0:
                        self.current_bid = float(tick.bid)
                        self.current_ask = float(tick.ask)
                    else:
                        self.current_bid = float(last_row["close"])
                        self.current_ask = self.current_bid + (self.params.spread_pips * self.params.pip_size)
                    self.executor.update_prices(self.symbol, self.current_bid, self.current_ask)
                    logger.info(f"Loaded {len(df)} candles live dari MT5: {self.symbol} {self.timeframe}")
                    return
            except Exception as e:
                logger.warning(f"Gagal mengambil rates dari MT5: {e}")

        if isinstance(self.executor, MT5Executor) and not supported:
            self.last_diag = f"Simbol {self.symbol} tidak tersedia di broker Finex MT5 (menampilkan data simulasi lokal; pindah ke EURUSD/GBPUSD/XAUUSD/US100 untuk data live)."

        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["time"] = pd.to_datetime(df["time"])
            self.df_candles = df.set_index("time").sort_index()
        else:
            default_csv = f"data/{self.symbol}_{self.timeframe}.csv"
            if os.path.exists(default_csv):
                df = pd.read_csv(default_csv)
                df["time"] = pd.to_datetime(df["time"])
                self.df_candles = df.set_index("time").sort_index()
            else:
                from generate_sample_data import generate_market_data
                df = generate_market_data(self.symbol, self.timeframe, bars=1000)
                df["time"] = pd.to_datetime(df["time"])
                self.df_candles = df.set_index("time").sort_index()

        if len(self.df_candles) > 0:
            last_row = self.df_candles.iloc[-1]
            self.last_candle_time = self.df_candles.index[-1]
            self.current_bid = float(last_row["close"])
            spread = self.params.spread_pips * self.params.pip_size
            self.current_ask = self.current_bid + spread
            self.executor.update_prices(self.symbol, self.current_bid, self.current_ask)

    def _simulate_tick(self) -> None:
        """Simulasi pergerakan tick real-time saat berjalan di Paper mode."""
        import random
        sym = self.symbol.upper()
        if "BTC" in sym:
            vol = random.uniform(5.0, 20.0)
        elif "ETH" in sym:
            vol = random.uniform(0.5, 2.5)
        elif "XAU" in sym or "GOLD" in sym:
            vol = random.uniform(0.15, 0.60)
        elif "XAG" in sym:
            vol = random.uniform(0.02, 0.08)
        elif "JPY" in sym:
            vol = random.uniform(0.01, 0.04)
        else:
            vol = self.params.pip_size * random.uniform(0.8, 2.5)

        delta = random.choice([-1, 1]) * vol
        self.current_bid = max(0.0001, self.current_bid + delta)
        spread = self.params.spread_pips * self.params.pip_size
        self.current_ask = self.current_bid + spread
        self.executor.update_prices(self.symbol, self.current_bid, self.current_ask)

    def _market_closed(self, symbol: Optional[str] = None) -> bool:
        sym = (symbol or self.symbol).upper()
        if "BTC" in sym or "ETH" in sym:
            return False
        t = datetime.now(timezone.utc)
        return t.weekday() == 5 or (t.weekday() == 6 and t.hour < 21) or (t.weekday() == 4 and t.hour >= 22)

    def _get_symbol_market_data(self, sym: str) -> tuple[float, float, Optional[pd.DataFrame]]:
        """Ambil bid, ask, dan DataFrame lilin terkini untuk simbol tertentu (mendukung multi-pair autonomous scanning)."""
        sym_clean = sym.upper()
        if sym_clean == self.symbol.upper():
            return self.current_bid, self.current_ask, self.df_candles

        pip = get_symbol_pip_size(sym_clean)
        default_price = float(SYMBOL_SPECS.get(sym_clean, {}).get("default_price", 1.0))
        spread_pips = float(SYMBOL_SPECS.get(sym_clean, {}).get("spread_pips", 1.0))

        if isinstance(self.executor, MT5Executor):
            try:
                with self.executor._lock:
                    mt5 = self.executor.mt5
                    if not mt5.symbol_select(sym_clean, True):
                        return 0.0, 0.0, None
                    tick = mt5.symbol_info_tick(sym_clean)
                    if not tick or tick.bid <= 0:
                        return 0.0, 0.0, None
                    bid = float(tick.bid)
                    ask = float(tick.ask)

                    tf_map = {
                        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                        "D1": mt5.TIMEFRAME_D1, "W1": getattr(mt5, "TIMEFRAME_W1", mt5.TIMEFRAME_D1),
                        "MN": getattr(mt5, "TIMEFRAME_MN1", mt5.TIMEFRAME_D1),
                    }
                    tf_val = tf_map.get(self.timeframe, mt5.TIMEFRAME_M5)
                    rates = mt5.copy_rates_from_pos(sym_clean, tf_val, 0, 80)
                    if rates is not None and len(rates) > 0:
                        raw_df = pd.DataFrame(rates)
                        raw_df["time"] = pd.to_datetime(raw_df["time"], unit="s")
                        df = raw_df.set_index("time").sort_index()
                        self.executor.update_prices(sym_clean, bid, ask)
                        return bid, ask, df
                    return bid, ask, None
            except Exception as e:
                logger.debug(f"Gagal get data pasar simbol {sym_clean}: {e}")
                return 0.0, 0.0, None

        # Paper Simulator Fallback
        bid = default_price
        ask = default_price + (spread_pips * pip)
        return bid, ask, None

    def _features(self, df, side, sl, tp, price, spread_pips) -> dict:
        p = self.params
        h, l, c = df["high"], df["low"], df["close"]
        atr = float((pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)).tail(14).mean()) or 1e-9
        last = df.iloc[-1]
        rng = max(float(last["high"] - last["low"]), 1e-12)
        wick = (min(last["open"], last["close"]) - last["low"] if side == 1 else last["high"] - max(last["open"], last["close"])) / rng
        risk = abs(price - sl)
        hr = df.index[-1].hour

        zone_depth = 0.5
        try:
            st = align_htf(df, htf_structure(resample_ohlc(df, p.htf_rule), p.pivot_k)).iloc[-1]
            if side == 1 and not pd.isna(st.get("buy_H")) and not pd.isna(st.get("buy_L")):
                leg = float(st["buy_H"] - st["buy_L"])
                if leg > 0:
                    zone_depth = max(-0.5, min(1.5, float(st["buy_H"] - price) / leg))
            elif side == -1 and not pd.isna(st.get("sell_H")) and not pd.isna(st.get("sell_L")):
                leg = float(st["sell_H"] - st["sell_L"])
                if leg > 0:
                    zone_depth = max(-0.5, min(1.5, float(price - st["sell_L"]) / leg))
        except Exception:
            pass

        return {
            "trend": float(side),
            "zone_depth": float(zone_depth),
            "wick": float(wick),
            "rr": abs(tp - price) / risk if risk else 0.0,
            "sl_atr": risk / atr,
            "hour_sin": math.sin(2 * math.pi * hr / 24),
            "hour_cos": math.cos(2 * math.pi * hr / 24),
            "spread": float(spread_pips),
        }

    def _learn_from_history(self) -> None:
        if not self.pending_features:
            return
        for t in self.executor.get_trade_history():
            tk = t.get("ticket")
            f = self.pending_features.pop(tk, None)
            if f is not None:
                is_explore = tk in self.pending_explore
                self.pending_explore.discard(tk)
                lr = self.learner_explore if is_explore else self.learner
                lr.update(f, t.get("pnl", 0) > 0)
                self.order_manager.log(f"BELAJAR{' (latihan)' if is_explore else ''}: trade #{tk} {'MENANG' if t['pnl'] > 0 else 'KALAH'} -> model diperbarui ({lr.n} sampel)", "INFO")

    def _on_new_candle_closed(self, closed_bar: pd.Series, df: Optional[pd.DataFrame] = None) -> None:
        """Dipanggil saat candle tertutup. df HANYA berisi candle yang sudah tertutup (tanpa lookahead)."""
        df = self.df_candles if df is None else df
        digits = symbol_decimals(self.symbol)
        self.order_manager.log(f"Bar tertutup @ {closed_bar.name} | Close: {closed_bar['close']:.{digits}f}", "INFO")
        if self.order_manager.cooldown_remaining > 0:
            self.order_manager.cooldown_remaining -= 1
        self.bars_since_explore += 1
        for _t in self.executor.get_trade_history():      # hitung R harian dari trade yang sudah tertutup
            self.order_manager.register_closed_trade(_t)
        self._learn_from_history()

        if self.is_paused:
            return
        if isinstance(self.executor, MT5Executor) and not getattr(self, "symbol_supported", True):
            self.last_diag = f"Simbol {self.symbol} tidak didukung oleh broker Finex MT5. Pindah ke EURUSD, GBPUSD, XAUUSD, atau US100 untuk trading live."
            return
        if self._market_closed():
            self.last_diag = "Market Forex/Komoditas TUTUP (akhir pekan) - bot menunggu pembukaan Senin."
            self.order_manager.log(self.last_diag, "WARNING")
            return

        signals = generate_signals(df, self.params)
        last_sig = signals.iloc[-1]
        side = int(last_sig["side"])
        if side == 0:
            if getattr(self.params, "explore_mode", False):
                self._try_explore_entry(df)
                return
            st_df = align_htf(df, htf_structure(resample_ohlc(df, self.params.htf_rule), self.params.pivot_k))
            st = st_df.iloc[-1] if not st_df.empty else pd.Series({"trend": float("nan")})
            tr = {1: "NAIK", -1: "TURUN"}.get(int(st["trend"]) if st["trend"] == st["trend"] else 0, "RANGING (tidak ada HH+HL / LH+LL)")
            self.last_diag = f"Tidak ada setup | Tren {self.params.htf_rule}: {tr} | candle belum menyentuh zona Fib 50-78.6% dengan rejection" + (f" atau ditolak filter MA ({self.params.ma_type} {self.params.ma_fast}/{self.params.ma_slow}, {self.params.ma_mode})." if getattr(self.params, "use_ma_filter", False) else ".")
            self.order_manager.log(self.last_diag, "INFO")
            return

        sl, tp = float(last_sig["sl"]), float(last_sig["tp"])
        price = self.current_ask if side == 1 else self.current_bid
        spread_pips = (self.current_ask - self.current_bid) / self.params.pip_size
        feats = self._features(df, side, sl, tp, price, spread_pips)
        prob = self.learner.predict(feats)
        if self.learner.ready and prob < THRESHOLD_GATE:
            self.last_diag = f"Sinyal {'BUY' if side == 1 else 'SELL'} DITOLAK model belajar (peluang menang {prob:.0%} < {THRESHOLD_GATE:.0%})."
            self.order_manager.log(self.last_diag, "WARNING")
            return
        pos = self.order_manager.evaluate_and_execute(
            symbol=self.symbol, side=side, sl=sl, tp=tp, current_price=price, spread_pips=spread_pips)
        if pos:
            self.pending_features[pos.ticket] = feats
            self.bars_since_explore = 0
            self.last_diag = f"ENTRY {'BUY' if side == 1 else 'SELL'} dieksekusi (peluang model {prob:.0%})."

    def _try_explore_entry(self, df: pd.DataFrame) -> None:
        """Entry latihan saat tidak ada sinyal Fibonacci. Arah mengikuti tren HTF (dan MA bila aktif)."""
        p = self.params
        if self.get_mode() == "MT5_REAL":
            self.set_explore(False)
            self.order_manager.log("Akun REAL terdeteksi: mode latihan dimatikan otomatis.", "RISK_ALERT")
            return
        need = max(1, int(p.explore_every_bars))
        if self.bars_since_explore < need:
            self.last_diag = f"MODE LATIHAN: entry berikutnya dalam {need - self.bars_since_explore} candle."
            return

        info = align_htf(df, htf_structure(resample_ohlc(df, p.htf_rule), p.pivot_k)).iloc[-1]
        trend = int(info["trend"]) if info["trend"] == info["trend"] else 0
        mdir = int(ma_direction(ma_htf_frame(df, p).tail(1), p.ma_mode)[0]) if p.use_ma_filter else 0
        src = "struktur HTF"
        if trend == 0 and mdir != 0:
            trend, src = mdir, "MA"
        direction_pref = str(getattr(p, "scalping_direction", "BOTH") or "BOTH").upper()
        if direction_pref == "BUY_ONLY":
            trend, src = 1, "Mode BUY SAJA"
        elif direction_pref == "SELL_ONLY":
            trend, src = -1, "Mode SELL SAJA"
        else:
            if trend != 0 and mdir != 0 and trend != mdir:
                last = df.iloc[-1]
                trend = 1 if last["close"] > last["open"] else (-1 if last["close"] < last["open"] else 0)
                src = "momentum candle (resolusi arah DUA ARAH)"
            elif trend == 0:
                last = df.iloc[-1]
                trend = 1 if last["close"] > last["open"] else (-1 if last["close"] < last["open"] else 0)
                src = "momentum candle terakhir"
        if trend == 0:
            return

        h, l, c = df["high"], df["low"], df["close"]
        atr = float(pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1).tail(14).mean())
        if not atr > 0:
            return
        pip = p.pip_size
        dist = min(max(p.explore_sl_atr * atr, p.min_sl_pips * pip * 1.001), p.max_sl_pips * pip * 0.999)
        price = self.current_ask if trend == 1 else self.current_bid
        sl = price - trend * dist
        tp = price + trend * dist * max(p.explore_rr, p.min_rr + 0.01)
        spread_pips = (self.current_ask - self.current_bid) / pip
        feats = self._features(df, trend, sl, tp, price, spread_pips)
        prob = self.learner_explore.predict(feats)   # hanya dicatat; mode latihan tidak memfilter
        pos = self.order_manager.evaluate_and_execute(
            symbol=self.symbol, side=trend, sl=sl, tp=tp, current_price=price, spread_pips=spread_pips)
        if pos:
            self.pending_features[pos.ticket] = feats
            self.pending_explore.add(pos.ticket)
            self.bars_since_explore = 0
            self.last_diag = f"ENTRY LATIHAN {'BUY' if trend == 1 else 'SELL'} (arah dari {src}; peluang model latihan {prob:.0%})."
        else:
            self.last_diag = "MODE LATIHAN: entry ditahan pengaman (spread, batas posisi, atau circuit breaker). Lihat log."

    def _compute_active_signal(
        self,
        df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        technique: Optional[str] = None,
        _is_nested_all: bool = False,
        closed_only: bool = True,
    ) -> dict:
        """Hitung sinyal rekomendasi live saat ini (Arah BUY/SELL, Entry, TP, SL, Confidence). Mendukung multi-pair."""
        sym = (symbol or self.symbol).upper()
        raw_df = self.df_candles if (df is None and sym == self.symbol) else df

        # Anti-Repainting: jika data menyertakan candle yang sedang berjalan (bar 0),
        # evaluasi indikator teknikal murni pada candle yang telah selesai tutup.
        eval_df = raw_df.iloc[:-1] if (closed_only and raw_df is not None and len(raw_df) >= 21) else raw_df

        digits = symbol_decimals(sym)
        pip = get_symbol_pip_size(sym)
        target_usd = float(getattr(self.params, "target_profit_usd", 5.0) or 5.0)
        # Rasio RR proporsional (risiko SL dibatasi maksimal 1.0x dari target TP agar 1 menang menutup 1 rugi)
        target_loss_usd = min(target_usd * 1.0, float(getattr(self.params, "target_loss_usd", target_usd) or target_usd))

        # Cek saldo & margin bebas akun untuk adaptasi lot modal kecil otomatis
        try:
            acc_info = self.executor.get_account_info()
            equity_val = float(acc_info.get("equity", 10.0) or 10.0)
            free_margin = float(acc_info.get("free_margin", equity_val) or equity_val)
        except Exception:
            equity_val = 10.0
            free_margin = 10.0

        user_lot = float(getattr(self.params, "scalping_lot", 0.1) or 0.1)
        lot = user_lot

        # Pip value estimasi untuk simbol ini
        pip_val = estimate_pip_value(sym, pip)
        dollar_per_pip = pip_val * lot

        tp_pips = max(2.5, target_usd / dollar_per_pip if dollar_per_pip > 0 else 10.0)
        sl_pips = max(2.5, target_loss_usd / dollar_per_pip if dollar_per_pip > 0 else 10.0)

        # Tentukan bid dan ask pasar
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            cur_bid, cur_ask = bid, ask
        elif sym == self.symbol:
            cur_bid, cur_ask = self.current_bid, self.current_ask
        else:
            cur_bid = float(raw_df["close"].iloc[-1]) if raw_df is not None and len(raw_df) > 0 else float(SYMBOL_SPECS.get(sym, {}).get("default_price", 1.0))
            cur_ask = cur_bid + (float(SYMBOL_SPECS.get(sym, {}).get("spread_pips", self.params.spread_pips)) * pip)

        spread_p = max(0.1, round((cur_ask - cur_bid) / pip, 2)) if pip > 0 else 1.0

        direction_mode = str(getattr(self.params, "scalping_direction", "BOTH") or "BOTH").upper()
        active_tech = "EMA_TREND"
        tech_meta = AVAILABLE_TECHNIQUES.get("EMA_TREND", {})

        # Fail-closed: jika data candle belum mencukupi (minimal 20 bar selesai), tolak entri!
        if eval_df is None or len(eval_df) < 20:
            count_bars = len(eval_df) if eval_df is not None else 0
            return {
                "action": "WAIT",
                "conf": 0.0,
                "reason": f"[{tech_meta.get('short_name', 'EMA Trend')}] Data candle belum mencukupi ({count_bars}/20 bar)",
                "symbol": sym,
                "technique": "EMA_TREND",
                "entry": cur_ask,
                "sl": 0.0,
                "tp": 0.0,
                "sl_pips": 0.0,
                "tp_pips": 0.0,
                "lot": 0.0,
                "risk_usd": 0.0,
                "target_usd": 0.0,
                "spread_pips": spread_p,
            }

        action = "WAIT"
        conf = 50.0
        reason = f"[{tech_meta.get('short_name', 'EMA Trend')}] Analisis tren EMA 8/21/50 aktif"
        contributing_techs = ["EMA_TREND"]
        atr = 5.0 * pip
        last_h = cur_ask + 5.0 * pip
        last_l = cur_bid - 5.0 * pip
        ema50 = cur_bid
        news_info_local = None

        c = eval_df["close"]
        o = eval_df["open"]
        h = eval_df["high"]
        l = eval_df["low"]

        last_c = float(c.iloc[-1])
        last_o = float(o.iloc[-1])
        last_h = float(h.iloc[-1])
        last_l = float(l.iloc[-1])

        candle_body = last_c - last_o
        candle_range = max(1e-9, last_h - last_l)
        upper_wick = last_h - max(last_c, last_o)
        lower_wick = min(last_c, last_o) - last_l

        # 1. EXPONENTIAL MOVING AVERAGE (EMA 8 / 21 / 50 RIBBON)
        ema8_series = c.ewm(span=8, adjust=False).mean()
        ema21_series = c.ewm(span=21, adjust=False).mean()
        ema50_series = c.ewm(span=50, adjust=False).mean() if len(c) >= 50 else ema21_series

        ema8 = float(ema8_series.iloc[-1])
        ema21 = float(ema21_series.iloc[-1])
        ema50 = float(ema50_series.iloc[-1])

        ema8_prev = float(ema8_series.iloc[-2]) if len(ema8_series) >= 2 else ema8
        ema21_prev = float(ema21_series.iloc[-2]) if len(ema21_series) >= 2 else ema21
        ema50_prev = float(ema50_series.iloc[-2]) if len(ema50_series) >= 2 else ema50

        # Slope (Kemiringan Garis) dalam Pips
        ema8_slope = (ema8 - ema8_prev) / pip
        ema21_slope = (ema21 - ema21_prev) / pip

        # True Range & ATR (14 Periode)
        tr_series = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr_series.tail(14).mean()) if len(tr_series) >= 14 else max(candle_range, 5.0 * pip)
        atr = max(atr, 2.0 * pip)

        # 2. RELATIVE STRENGTH INDEX (RSI 14) - Filter Jenuh Beli / Jual (Exhaustion Gate)
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_val = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50.0

        # 3. AVERAGE DIRECTIONAL INDEX (ADX 14) - Filter Anti-Sideways / Anti-Chop Gate
        up_move = h.diff()
        down_move = -l.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_smooth = tr_series.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100.0 * (pd.Series(plus_dm, index=eval_df.index).ewm(alpha=1/14, adjust=False).mean() / (tr_smooth + 1e-9))
        minus_di = 100.0 * (pd.Series(minus_dm, index=eval_df.index).ewm(alpha=1/14, adjust=False).mean() / (tr_smooth + 1e-9))
        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        adx_series = dx.ewm(alpha=1/14, adjust=False).mean()
        adx_val = float(adx_series.iloc[-1]) if len(adx_series) > 0 else 25.0

        # 4. HORIZONTAL SWING S/R (High/Low 20 Candle) - Room-to-Move Filter
        swing_h20 = float(h.iloc[-21:-1].max()) if len(h) >= 21 else (last_c + 10.0 * pip)
        swing_l20 = float(l.iloc[-21:-1].min()) if len(l) >= 21 else (last_c - 10.0 * pip)
        room_to_res = swing_h20 - last_c
        room_to_sup = last_c - swing_l20

        # Kondisi Susunan Ribbon (Alignment Arus Utama)
        is_bull_align = (ema8 > ema21 and ema21 > ema50)
        is_bear_align = (ema8 < ema21 and ema21 < ema50)

        # Filter Konteks Makro M15 (EMA 50 M15 setara ~EMA 150 M5)
        ema_m15_50 = float(c.ewm(span=min(len(c), 150), adjust=False).mean().iloc[-1])
        m15_bullish = (last_c >= ema_m15_50)
        m15_bearish = (last_c <= ema_m15_50)

        # Crossover Dinamis
        fresh_golden_cross = (ema8 >= ema21 and ema8_prev < ema21_prev)
        fresh_death_cross = (ema8 <= ema21 and ema8_prev > ema21_prev)

        # Jarak Harga ke EMA 8 (Filter Overextension / Anti-Chasing Pucuk & Lembah)
        overextended_bull = (last_c - ema8) > (2.0 * atr)
        overextended_bear = (ema8 - last_c) > (2.0 * atr)

        # Pullback & Rejection Zone (Dynamic S/R antara EMA 8 dan EMA 21)
        bull_in_pocket = (last_l <= ema8 + 1.2 * pip) and (last_c >= ema21 - 2.0 * pip)
        bear_in_pocket = (last_h >= ema8 - 1.2 * pip) and (last_c <= ema21 + 2.0 * pip)

        # Konfirmasi Kualitas Candle: Minimal Range 0.45 ATR & Rejection Wick >= 35% atau Engulfing Kuat (Body >= 50%)
        valid_candle_size = candle_range >= (0.45 * atr)
        is_bull_rejection = valid_candle_size and (
            (lower_wick >= 0.35 * candle_range) or
            (candle_body > 0 and (last_c - last_o) >= 0.50 * candle_range and last_c > ema8)
        )
        is_bear_rejection = valid_candle_size and (
            (upper_wick >= 0.35 * candle_range) or
            (candle_body < 0 and (last_o - last_c) >= 0.50 * candle_range and last_c < ema8)
        )

        if is_bull_align or fresh_golden_cross:
            if not m15_bullish:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter Makro M15: Harga di bawah EMA 50 M15 ({ema_m15_50:.{digits}f}). Dilarang BUY melawan tren makro (Hindari Bull-Trap)."
            elif adx_val < 20.0:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter Anti-Chop: ADX rendah ({adx_val:.1f} < 20). Pasar sedang sideways / tanpa tren. Dilarang entry MA untuk cegah whipsaw."
            elif ema21_slope < 0.20:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter Momentum (Slope): Sudut EMA 21 mendatar ({ema21_slope:+.2f}p). Dorongan tren naik belum cukup kuat."
            elif rsi_val > 68.0:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter RSI Exhaustion: RSI jenuh beli ({rsi_val:.1f} > 68). Risiko tinggi buying the top / koreksi tajam."
            elif room_to_res < (1.0 * atr):
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter Ruang Gerak S/R: Harga terlalu dekat dengan Resistance Swing High ({swing_h20:.{digits}f}, sisa {room_to_res/pip:.1f}p). Ruang gerak sempit."
            elif overextended_bull:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ EMA Trend: Harga overextended (+{(last_c - ema8)/pip:.1f} pip dari EMA 8). Menunggu pullback sehat ke zona EMA 8/21 sebelum BUY."
            elif bull_in_pocket and is_bull_rejection:
                action = "BUY"
                conf = 50.0 + (15.0 if adx_val >= 25.0 else 10.0) + (15.0 if 45.0 <= rsi_val <= 60.0 else 10.0) + 10.0 + (10.0 if m15_bullish else 0.0)
                reason = f"📈 EMA Trend BUY [A+ Retest]: Pullback ke Dynamic Support (EMA 8/21) & Rejection Bullish ({ema8:.{digits}f}) | ADX kuat ({adx_val:.1f}) & RSI sehat ({rsi_val:.1f})."
            elif fresh_golden_cross and candle_body > (0.45 * candle_range) and last_c > ema8 and ema8_slope > 0.35:
                action = "BUY"
                conf = 50.0 + (15.0 if adx_val >= 25.0 else 10.0) + (15.0 if 45.0 <= rsi_val <= 60.0 else 10.0) + 5.0 + (10.0 if m15_bullish else 0.0)
                reason = f"🚀 EMA Trend BUY [Golden Cross]: EMA 8 menembus ke atas EMA 21 dengan candle bullish impulsif ({last_c:.{digits}f}) | ADX: {adx_val:.1f}."
            else:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ EMA Trend: Harga belum menyentuh Dynamic Support EMA 8/21 atau belum ada candle rejection valid (Disiplin WAIT)."

        elif is_bear_align or fresh_death_cross:
            if not m15_bearish:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter Makro M15: Harga di atas EMA 50 M15 ({ema_m15_50:.{digits}f}). Dilarang SELL melawan tren makro (Hindari Bear-Trap)."
            elif adx_val < 20.0:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter Anti-Chop: ADX rendah ({adx_val:.1f} < 20). Pasar sedang sideways / tanpa tren. Dilarang entry MA untuk cegah whipsaw."
            elif ema21_slope > -0.20:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter Momentum (Slope): Sudut EMA 21 mendatar ({ema21_slope:+.2f}p). Dorongan tren turun belum cukup kuat."
            elif rsi_val < 32.0:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter RSI Exhaustion: RSI jenuh jual ({rsi_val:.1f} < 32). Risiko tinggi selling the bottom / pantulan tajam."
            elif room_to_sup < (1.0 * atr):
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ Filter Ruang Gerak S/R: Harga terlalu dekat dengan Support Swing Low ({swing_l20:.{digits}f}, sisa {room_to_sup/pip:.1f}p). Ruang gerak sempit."
            elif overextended_bear:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ EMA Trend: Harga overextended (-{(ema8 - last_c)/pip:.1f} pip dari EMA 8). Menunggu koreksi naik ke zona EMA 8/21 sebelum SELL."
            elif bear_in_pocket and is_bear_rejection:
                action = "SELL"
                conf = 50.0 + (15.0 if adx_val >= 25.0 else 10.0) + (15.0 if 40.0 <= rsi_val <= 55.0 else 10.0) + 10.0 + (10.0 if m15_bearish else 0.0)
                reason = f"📉 EMA Trend SELL [A+ Retest]: Pullback ke Dynamic Resistance (EMA 8/21) & Rejection Bearish ({ema8:.{digits}f}) | ADX kuat ({adx_val:.1f}) & RSI sehat ({rsi_val:.1f})."
            elif fresh_death_cross and candle_body < -(0.45 * candle_range) and last_c < ema8 and ema8_slope < -0.35:
                action = "SELL"
                conf = 50.0 + (15.0 if adx_val >= 25.0 else 10.0) + (15.0 if 40.0 <= rsi_val <= 55.0 else 10.0) + 5.0 + (10.0 if m15_bearish else 0.0)
                reason = f"🩸 EMA Trend SELL [Death Cross]: EMA 8 menembus ke bawah EMA 21 dengan candle bearish impulsif ({last_c:.{digits}f}) | ADX: {adx_val:.1f}."
            else:
                action = "WAIT"
                conf = 0.0
                reason = f"⏳ EMA Trend: Harga belum menyentuh Dynamic Resistance EMA 8/21 atau belum ada candle rejection valid (Disiplin WAIT)."

        else:
            action = "WAIT"
            conf = 0.0
            reason = f"⏳ EMA Trend: Moving Average konsolidasi/sideways (EMA 8: {ema8:.{digits}f}, EMA 21: {ema21:.{digits}f}, EMA 50: {ema50:.{digits}f}). Menunggu pembentukan tren searah."

        # Integrasi Model Prediktif EURUSD (Regresi Linear Kuantitatif)
        if sym == "EURUSD" and df is not None and len(df) >= 50:
            try:
                state_path = os.path.join(os.path.dirname(__file__), "data", "eurusd_model_state.json")
                if os.path.exists(state_path):
                    with open(state_path, "r", encoding="utf-8") as f:
                        m_st = json.load(f)
                    c_vals = eval_df["close"].astype(float).values
                    o_vals = eval_df["open"].astype(float).values
                    h_vals = eval_df["high"].astype(float).values
                    l_vals = eval_df["low"].astype(float).values
                    v_vals = eval_df["tick_volume"].astype(float).values if "tick_volume" in eval_df.columns else np.ones(len(eval_df))

                    m8 = float(pd.Series(c_vals).rolling(8).mean().iloc[-1])
                    m21 = float(pd.Series(c_vals).rolling(21).mean().iloc[-1])
                    m50 = float(pd.Series(c_vals).rolling(50).mean().iloc[-1])
                    m200 = float(pd.Series(c_vals).rolling(min(200, len(c_vals))).mean().iloc[-1])

                    s8 = float(pd.Series(c_vals).rolling(8).std().iloc[-1])
                    s21 = float(pd.Series(c_vals).rolling(21).std().iloc[-1])
                    s50 = float(pd.Series(c_vals).rolling(50).std().iloc[-1])

                    rng_e = max(float(h_vals[-1] - l_vals[-1]), 1e-6)

                    f_v = [
                        float(o_vals[-1]), float(c_vals[-1]), float(h_vals[-1]), float(l_vals[-1]), float(v_vals[-1]),
                        (m8 / (m21 + 1e-9)) - 1.0, (m8 / (m50 + 1e-9)) - 1.0, (m21 / (m50 + 1e-9)) - 1.0, (m50 / (m200 + 1e-9)) - 1.0,
                        (float(c_vals[-1]) - m8) / pip, (float(c_vals[-1]) - m21) / pip, (float(c_vals[-1]) - m50) / pip,
                        s8 / (s21 + 1e-9), s21 / (s50 + 1e-9), s21 / pip,
                        float((c_vals[-1] - c_vals[-2]) / (c_vals[-2] + 1e-9)),
                        float((c_vals[-1] - c_vals[-6]) / (c_vals[-6] + 1e-9)) if len(c_vals) >= 6 else 0.0,
                        float((c_vals[-1] - c_vals[-22]) / (c_vals[-22] + 1e-9)) if len(c_vals) >= 22 else 0.0,
                        float((c_vals[-1] - c_vals[-51]) / (c_vals[-51] + 1e-9)) if len(c_vals) >= 51 else 0.0,
                        float(v_vals[-1] / (pd.Series(v_vals).rolling(21).mean().iloc[-1] + 1e-9)),
                        (float(c_vals[-1]) - float(o_vals[-1])) / rng_e,
                        (float(h_vals[-1]) - max(float(c_vals[-1]), float(o_vals[-1]))) / rng_e,
                        (min(float(c_vals[-1]), float(o_vals[-1])) - float(l_vals[-1])) / rng_e,
                    ]

                    m_mean = np.array(m_st["scaler_mean"])
                    m_scale = np.array(m_st["scaler_scale"])
                    m_w = np.array(m_st["weights"])
                    m_b = float(m_st["intercept"])

                    drift = float(np.dot((np.array(f_v) - m_mean) / (m_scale + 1e-9), m_w) + m_b)

                    if action == "BUY":
                        if drift < -2.0:
                            action = "WAIT"
                            conf = 0.0
                            reason = f"⏳ Filter Model EURUSD: Drift negatif ({drift:+.1f}p). Dilarang BUY melawan model regresi."
                        elif drift > 1.2:
                            reason += f" | Model Regresi EURUSD: Prediksi Drift Positif ({drift:+.1f}p)"
                    elif action == "SELL":
                        if drift > 2.0:
                            action = "WAIT"
                            conf = 0.0
                            reason = f"⏳ Filter Model EURUSD: Drift positif ({drift:+.1f}p). Dilarang SELL melawan model regresi."
                        elif drift < -1.2:
                            reason += f" | Model Regresi EURUSD: Prediksi Drift Negatif ({drift:+.1f}p)"
            except Exception as e_eur:
                logger.debug(f"Evaluasi model EURUSD dilewati: {e_eur}")

        # Filter Pembatasan Arah Pengguna
        if action in ("BUY", "SELL"):
            if direction_mode == "BUY_ONLY" and action == "SELL":
                action = "WAIT"
                reason = f"Sinyal SELL diabaikan karena mode diset ke BUY SAJA"
            elif direction_mode == "SELL_ONLY" and action == "BUY":
                action = "WAIT"
                reason = f"Sinyal BUY diabaikan karena mode diset ke SELL SAJA"

            # PILAR 1: Stop Loss Tempat Merubah View (Structural Invalidation Gate)
            loss_info = getattr(self, "last_loss_by_symbol", {}).get(sym)
            if loss_info and action == loss_info.get("side"):
                l_side = loss_info.get("side")
                l_px = float(loss_info.get("price", 0.0) or 0.0)
                l_ts = float(loss_info.get("ts", 0.0) or 0.0)
                l_elapsed = time.time() - l_ts
                if l_elapsed < 600.0:
                    action = "WAIT"
                    conf = 45.0
                    rem_m = int((600.0 - l_elapsed) / 60) + 1
                    reason = f"🛑 PILAR 1 (Stop Loss Tempat Merubah View): View {l_side} dibatalkan karena kena SL di {l_px:.2f}. Cooldown disiplin {rem_m}m tersisa."
                elif l_side == "BUY" and last_c < (l_px + 15.0 * pip):
                    action = "WAIT"
                    conf = 48.0
                    reason = f"🛑 PILAR 1 (Stop Loss Tempat Merubah View): BUY dibatalkan karena kena SL di {l_px:.2f}. Struktur belum pulih di atas {l_px + 15.0*pip:.2f}."
                elif l_side == "SELL" and last_c > (l_px - 15.0 * pip):
                    action = "WAIT"
                    conf = 48.0
                    reason = f"🛑 PILAR 1 (Stop Loss Tempat Merubah View): SELL dibatalkan karena kena SL di {l_px:.2f}. Struktur belum pulih di bawah {l_px - 15.0*pip:.2f}."
        # Terapkan filter arah jika dikonfigurasi
        if direction_mode == "BUY_ONLY" and action == "SELL":
            action = "WAIT"
            reason = "Disiplin Filter Arah: Mode BUY_ONLY aktif, sinyal SELL diabaikan."
        elif direction_mode == "SELL_ONLY" and action == "BUY":
            action = "WAIT"
            reason = "Disiplin Filter Arah: Mode SELL_ONLY aktif, sinyal BUY diabaikan."

        # Kalkulasi Entry, SL, TP Proporsional (RR 1 : 1.6+ & Proteksi Noise Emas)
        if action == "BUY":
            entry = cur_ask
            sl_anchor = min(ema50 - 2.0 * pip, last_l - 2.5 * pip)
            sl_pips = round(min(65.0, max(35.0, max(1.2 * atr / pip, (entry - sl_anchor) / pip))), 1)
            tp_pips = round(max(sl_pips * 1.6, 50.0), 1)
            tp = round(entry + tp_pips * pip, digits)
            sl = round(entry - sl_pips * pip, digits)
            risk_usd = round(sl_pips * dollar_per_pip, 2)
            target_usd = round(tp_pips * dollar_per_pip, 2)
        elif action == "SELL":
            entry = cur_bid
            sl_anchor = max(ema50 + 2.0 * pip, last_h + 2.5 * pip)
            sl_pips = round(min(65.0, max(35.0, max(1.2 * atr / pip, (sl_anchor - entry) / pip))), 1)
            tp_pips = round(max(sl_pips * 1.6, 50.0), 1)
            tp = round(entry - tp_pips * pip, digits)
            sl = round(entry + sl_pips * pip, digits)
            risk_usd = round(sl_pips * dollar_per_pip, 2)
            target_usd = round(tp_pips * dollar_per_pip, 2)
        else:
            entry = cur_ask if cur_ask > 0 else (cur_bid if cur_bid > 0 else 1.0)
            tp = 0.0
            sl = 0.0
            sl_pips = 0.0
            tp_pips = 0.0
            risk_usd = 0.0
            target_usd = 0.0

        # Penyesuaian Lot dan Margin Bebas (Anti NO MONEY)
        specs = self.executor.get_symbol_trade_specs(sym, pip, entry)
        min_l = specs.get("min_lot", 0.01)
        step_l = specs.get("lot_step", 0.01)
        max_l = specs.get("max_lot", 100.0)

        lot = float(getattr(self.params, "scalping_lot", 0.1) or 0.1)
        if hasattr(self.executor, "get_required_margin"):
            try:
                req_m = self.executor.get_required_margin(sym, action, lot, entry)
                min_m = self.executor.get_required_margin(sym, action, min_l, entry)
                if free_margin > 0 and free_margin < min_m and action in ("BUY", "SELL"):
                    action = "WAIT"
                    conf = 50.0
                    reason = f"💰 Saldo / Margin Menipis: Margin bebas (${free_margin:.2f}) di bawah syarat margin {min_l} lot {sym} (${min_m:.2f}). Tunggu posisi aktif selesai atau tambah saldo."
                elif free_margin > 0 and req_m > free_margin * 0.90:
                    lot = min_l
            except Exception:
                pass

        if step_l > 0:
            lot = round(round(lot / step_l) * step_l, 2)
        lot = max(min_l, min(max_l, lot))
        calculated_lot = lot

        session_info = get_current_market_session()
        sig = {
            "symbol": sym,
            "market_session": session_info.get("session", "London/NY"),
            "market_session_name": session_info.get("name", "Sesi London/NY"),
            "market_session_active": session_info.get("is_active_session", True),
            "market_session_icon": session_info.get("icon", "🏛️"),
            "market_session_hours": session_info.get("hours_utc", ""),
            "decision": action,
            "action": action,
            "entry_price": round(entry, digits),
            "stop_loss": sl,
            "take_profit": tp,
            "tp_price": tp,
            "sl_price": sl,
            "tp_pips": round(tp_pips, 1),
            "sl_pips": round(sl_pips, 1),
            "target_usd": target_usd,
            "risk_usd": risk_usd,
            "lot": lot,
            "calculated_lot": calculated_lot,
            "confidence": round(conf, 1) if action in ("BUY", "SELL") else 0.0,
            "setup_score": round(conf, 1) if action in ("BUY", "SELL") else 0.0,
            "reason": reason,
            "technique": "EMA_TREND",
            "contributing_techniques": ["EMA_TREND"],
            "active_technique_id": "EMA_TREND",
            "technique_name": tech_meta.get("name", "Trend Following Moving Average (EMA 8 / 21 / 50 Ribbon)"),
            "technique_icon": tech_meta.get("icon", "fa-chart-line"),
            "ema_ribbon": {
                "ema8": round(ema8, digits) if 'ema8' in locals() else round(entry, digits),
                "ema21": round(ema21, digits) if 'ema21' in locals() else round(entry, digits),
                "ema50": round(ema50, digits) if 'ema50' in locals() else round(entry, digits),
                "ema8_slope": round(ema8_slope, 2) if 'ema8_slope' in locals() else 0.0,
                "ema21_slope": round(ema21_slope, 2) if 'ema21_slope' in locals() else 0.0,
                "alignment": "BULLISH" if ('is_bull_align' in locals() and is_bull_align) else ("BEARISH" if ('is_bear_align' in locals() and is_bear_align) else "SIDEWAYS"),
                "adx": round(adx_val, 1) if 'adx_val' in locals() else 25.0,
                "rsi": round(rsi_val, 1) if 'rsi_val' in locals() else 50.0,
                "trend_strength": "KUAT (ADX >= 20)" if ('adx_val' in locals() and adx_val >= 20.0) else "LEMAH / CHOP (ADX < 20)",
                "room_to_move_pips": round(room_to_res / pip if ('is_bull_align' in locals() and is_bull_align) else (room_to_sup / pip if ('is_bear_align' in locals() and is_bear_align) else 0.0), 1) if 'room_to_res' in locals() else 0.0,
            },
            "scalping_zone": {
                "upper": round(ema8, digits) if 'ema8' in locals() else round(entry, digits),
                "lower": round(ema21, digits) if 'ema21' in locals() else round(entry, digits),
                "in_zone": bool(bull_in_pocket if ('bull_in_pocket' in locals() and is_bull_align) else (bear_in_pocket if ('bear_in_pocket' in locals() and is_bear_align) else False)),
                "name": "Dynamic Pocket EMA 8/21 Golden Zone",
            },
            "market_view": getattr(self, "view_bias", "BULLISH" if ('is_bull_align' in locals() and is_bull_align) else ("BEARISH" if ('is_bear_align' in locals() and is_bear_align) else "NEUTRAL")),
            "view_shift_status": getattr(self, "view_shift_status", "NORMAL"),
            "time": datetime.now(timezone.utc).isoformat(),
            "trading_mode": getattr(self.params, "trading_mode", "scalping"),
            "scalping_direction": direction_mode,
            "gemini_news": None,
        }

        if sym == self.symbol:
            self.active_signal = sig
        return sig


    def evaluate_all_techniques(
        self,
        df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> dict:
        """Evaluasi status kesiapan (SIAP OPEN POSISI vs MENUNGGU) untuk ke-8 teknik trading secara live."""
        sym = (symbol or self.symbol).upper()
        cur_df = self.df_candles if (df is None and sym == self.symbol) else df
        res = {}
        for tech_id in AVAILABLE_TECHNIQUES.keys():
            try:
                sig = self._compute_active_signal(cur_df, symbol=sym, bid=bid, ask=ask, technique=tech_id)
                action = sig.get("action", "WAIT")
                is_ready = action in ("BUY", "SELL")
                res[tech_id] = {
                    "id": tech_id,
                    "action": action,              # "BUY", "SELL", atau "WAIT"
                    "is_ready": is_ready,          # True jika sudah ada sinyal BUY / SELL
                    "confidence": sig.get("confidence", 50.0),
                    "reason": sig.get("reason", "Menunggu konfirmasi setup..."),
                    "entry_price": sig.get("entry_price"),
                    "tp_price": sig.get("tp_price"),
                    "sl_price": sig.get("sl_price"),
                }
            except Exception as e:
                res[tech_id] = {
                    "id": tech_id,
                    "action": "WAIT",
                    "is_ready": False,
                    "confidence": 40.0,
                    "reason": f"Menunggu data pasar ({e})",
                }
        return res

    def _try_scalping_entry(self) -> None:
        """Pindai dan eksekusi entri scalping otomatis mandiri di multi-pair (EURUSD, XAUUSD, GBPUSD, US100)."""
        if not self.is_running or self.is_paused:
            return
        if getattr(self.params, "trading_mode", "scalping") != "scalping":
            return
        if self._market_closed():
            return

        # PILAR 0: Proteksi Berita High-Impact (News Shield Circuit Breaker: CPI, NFP, GDP, FOMC - Fail Closed)
        try:
            from news_radar import get_news_radar
            shield_st = get_news_radar().get_news_shield_status()
            if shield_st.get("is_shield_active", False):
                self.last_diag = shield_st.get("warning", "🚨 NEWS SHIELD AKTIF: Menahan order saat rilis High-Impact News.")
                return
        except Exception:
            pass

        # 0. Evaluasi CPI/PPI Auto-Sniper (Big Move Anticipator Berbasis Data Nowcasting)
        try:
            from cpi_predictor import get_cpi_predictor
            cpi_pred = get_cpi_predictor()
            if cpi_pred.sniper_enabled:
                # Evaluasi simbol yang relevan (XAUUSD dan EURUSD jika mode Semua Market, atau simbol aktif)
                eval_symbols = ["XAUUSD", "EURUSD"] if getattr(self, "symbol_mode", "single") == "multi" else [self.symbol]
                
                for target_sym in eval_symbols:
                    cur_bid, cur_ask = 0.0, 0.0
                    pip_sz = get_symbol_pip_size(target_sym)
                    
                    if target_sym == self.symbol:
                        cur_bid = self.current_bid
                        cur_ask = self.current_ask
                    elif hasattr(self, "executor") and hasattr(self.executor, "latest_prices"):
                        q = self.executor.latest_prices.get(target_sym, {})
                        cur_bid = q.get("bid", 0.0)
                        cur_ask = q.get("ask", 0.0)
                        
                    ref_p = cur_ask if cur_ask > 0 else (cur_bid if cur_bid > 0 else (4000.0 if "XAU" in target_sym else 1.1000))
                    
                    techs = None
                    if target_sym == self.symbol:
                        techs = {
                            "ema8": getattr(self, "last_ema8", 0.0),
                            "ema21": getattr(self, "last_ema21", 0.0),
                            "ema50": getattr(self, "last_ema50", 0.0),
                            "rsi": getattr(self, "last_rsi", 50.0),
                        }
                        
                    cpi_eval = cpi_pred.evaluate_cpi_auto_sniper(
                        current_price=ref_p,
                        pip_size=pip_sz,
                        symbol=target_sym,
                        current_bid=cur_bid,
                        current_ask=cur_ask,
                        technicals=techs,
                    )
                    
                    if cpi_eval.get("should_execute"):
                        all_pos = self.executor.get_positions()
                        has_cpi_pos_sym = any(
                            p.symbol == target_sym and (
                                getattr(p, "technique", "") in ("CPI_SNIPER", "PPI_SNIPER") or
                                "CPI_SNIPER" in str(getattr(p, "comment", "")).upper() or
                                "PPI_SNIPER" in str(getattr(p, "comment", "")).upper()
                            )
                            for p in all_pos
                        )
                        if not has_cpi_pos_sym and len(all_pos) < self.max_total_open_positions:
                            action_side = 1 if cpi_eval["action"] == "BUY" else -1
                            real_spread_pips = (cur_ask - cur_bid) / pip_sz if pip_sz > 0 else 1.0
                            pos = self.order_manager.execute_scalping(
                                symbol=target_sym,
                                side=action_side,
                                sl=cpi_eval["sl_price"],
                                tp=cpi_eval["tp_price"],
                                current_price=cpi_eval["entry_price"],
                                spread_pips=real_spread_pips,
                                lot=cpi_eval["lot"],
                                target_usd=cpi_eval["target_usd"],
                                technique="CPI_SNIPER",
                            )
                            if pos:
                                self.order_manager.log(
                                    f"🎯 AUTO MACRO SNIPER ({cpi_eval['action']}): {cpi_eval['lot']} lot {target_sym} @ {cpi_eval['entry_price']:.{symbol_decimals(target_sym)}f} | SL: {cpi_eval['sl_price']} | TP: {cpi_eval['tp_price']} | {cpi_eval['reason']}",
                                    "INFO",
                                )
                                return
        except Exception as e_cpi:
            logger.debug(f"CPI Auto-Sniper check skipped: {e_cpi}")

        # 8. Eksekusi Order Scalping Otomatis Berdasarkan MA EMA Ribbon

        active_sym = self.symbol
        now_ts = time.time()
        all_positions = self.executor.get_positions()
        mode_str = str(getattr(self.params, "scalping_max_open_mode", "AUTO")).upper()
        if mode_str in ("1", "2", "3", "4", "5"):
            max_total_open = int(mode_str)
        else:
            max_total_open = 5

        if len(all_positions) >= max_total_open:
            if active_sym == self.symbol and len(all_positions) > 0:
                self.last_diag = f"KUOTA MAKSIMAL ({len(all_positions)}/{max_total_open}): Batas kuota posisi aktif tercapai."
            return

        # Kaidah Mode Otomatis (Scale-In / Piramid saat Mendekati TP)
        if mode_str == "AUTO" and len(all_positions) > 0:
            can_scale_in, pyramid_reason = self._check_auto_pyramid_allowed(all_positions)
            if not can_scale_in:
                if active_sym == self.symbol:
                    self.last_diag = f"⏳ KUOTA OTOMATIS: {pyramid_reason}"
                return

        # PILAR 2: DISIPLIN SESI PASAR (Time Zone Gate)
        session_info = get_current_market_session()
        sess_mode = getattr(self.params, "session_filter_mode", "STRICT")
        use_sess = (sess_mode == "STRICT") and getattr(self.params, "use_session_filter", True)
        if use_sess and not session_info.get("is_active_session", True):
            if active_sym == self.symbol:
                self.last_diag = f"🌐 DISIPLIN SESI PASAR: {session_info.get('name', 'Sesi Asia')} ({session_info.get('status', 'WAIT')}). Auto-Scalping disiplin ditahan hingga likuiditas Sesi London/NY aktif."
            return

        # Urutan pemindaian simbol: Dahulukan simbol aktif (XAUUSD / EURUSD saja, TANPA US100)
        scan_symbols = [active_sym]
        if getattr(self, "multi_symbol_enabled", False):
            for s in getattr(self, "watchlist", ["XAUUSD", "EURUSD"]):
                if s not in scan_symbols and s != "US100":
                    scan_symbols.append(s)

        for sym in scan_symbols:
            if len(all_positions) >= max_total_open:
                break

            # 1. Cek kuota posisi per simbol
            sym_positions = [p for p in all_positions if p.symbol == sym]
            max_per_sym = 1 if mode_str == "1" else max_total_open
            if len(sym_positions) >= max_per_sym:
                continue

            # 2. Cek jeda minimal setelah transaksi tertutup pada simbol ini:
            # Jika trade sebelumnya ditutup PROFIT (termasuk Anti-Balik Minus / Break-Even), jeda hanya 15 detik
            # agar bot segera siap re-entry saat candle terbang lagi atau berbalik turun!
            # Jika trade sebelumnya kena SL (rugi), jeda 90s untuk stabilisasi pasar.
            last_closed = getattr(self, "last_closed_trade_ts_by_symbol", {}).get(sym, 0.0)
            is_profit_exit = getattr(self, "last_closed_was_profit_by_symbol", {}).get(sym, False)
            min_cooldown = 10.0 if is_profit_exit else 90.0
            if (now_ts - last_closed) < min_cooldown:
                rem_wait = int(min_cooldown - (now_ts - last_closed))
                if sym == self.symbol:
                    wait_label = "PASCA CUAN" if is_profit_exit else "PASCA TRADE"
                    self.last_diag = f"JEDA {wait_label} ({sym}): Menunggu {rem_wait}s stabilisasi pasar sebelum evaluasi setup baru."
                continue

            # 3. Cek interval scalping antar trade untuk simbol ini
            last_scalp = getattr(self, "last_scalp_by_symbol", {}).get(sym, 0.0)
            interval = int(getattr(self.params, "scalping_interval_sec", 60))
            if len(sym_positions) > 0 and (now_ts - last_scalp) < interval:
                continue

            # 4. Ambil data pasar real-time untuk sym
            sym_bid, sym_ask, sym_df = self._get_symbol_market_data(sym)
            if sym_bid <= 0 or sym_ask <= 0 or sym_df is None or len(sym_df) < 21:
                continue

            # 5. Hitung sinyal kuantitatif aktif untuk simbol ini (Evaluasi hanya pada candle yang telah selesai tutup)
            sig = self._compute_active_signal(sym_df, symbol=sym, bid=sym_bid, ask=sym_ask, closed_only=True)
            if not sig or sig.get("action") not in ("BUY", "SELL"):
                if sym == self.symbol and sig:
                    t_meta = AVAILABLE_TECHNIQUES.get(sig.get("technique", "EMA_TREND"), AVAILABLE_TECHNIQUES.get("EMA_TREND", {}))
                    self.last_diag = f"[{t_meta.get('short_name', 'EMA Trend')}] {sig.get('reason', 'Menunggu setup valid...')}"
                continue

            pip = get_symbol_pip_size(sym)

            # 6. PILAR 1: STOP LOSS TEMPAT MERUBAH VIEW & STRUCTURAL INVALIDATION
            loss_info = getattr(self, "last_loss_by_symbol", {}).get(sym)
            if loss_info:
                loss_elapsed = now_ts - loss_info.get("ts", 0)
                loss_side = loss_info.get("side")
                loss_px = float(loss_info.get("price", 0.0) or 0.0)
                swing_h = float(loss_info.get("swing_high", 0.0) or 0.0)
                swing_l = float(loss_info.get("swing_low", 0.0) or 0.0)
                cur_c = float(sym_df["close"].iloc[-1])

                if sig["action"] == loss_side:
                    # Cooldown 10 menit setelah kena SL untuk mendinginkan arus pasar
                    if loss_elapsed < 600.0:
                        rem_sec = int(600.0 - loss_elapsed)
                        if sym == self.symbol:
                            self.last_diag = f"🛑 STOP LOSS MERUBAH VIEW ({sym}): Order {sig['action']} DITAHAN ({rem_sec}s)! View {loss_side} telah invalid pasca SL."
                        continue

                    # Invalidation: Jika BUY kena SL, dilarang BUY lagi jika harga belum menembus Swing High sebelum breakdown
                    if loss_side == "BUY":
                        bench_h = max(swing_h, loss_px + 15.0 * pip) if swing_h > 0 else (loss_px + 15.0 * pip)
                        if cur_c < bench_h:
                            if sym == self.symbol:
                                self.last_diag = f"🛑 STOP LOSS MERUBAH VIEW ({sym}): BUY DITOLAK! View Bullish batal karena SL di {loss_px:.2f}. Wajib tunggu Breakout High ({bench_h:.2f})."
                            continue
                    elif loss_side == "SELL":
                        bench_l = min(swing_l, loss_px - 15.0 * pip) if swing_l > 0 else (loss_px - 15.0 * pip)
                        if cur_c > bench_l:
                            if sym == self.symbol:
                                self.last_diag = f"🛑 STOP LOSS MERUBAH VIEW ({sym}): SELL DITOLAK! View Bearish batal karena SL di {loss_px:.2f}. Wajib tunggu Breakdown Low ({bench_l:.2f})."
                            continue

            # 6b. PILAR 2: SCALPING MEMPUNYAI ZONA (One-Zone One-Trade Rule & Anti-Spam)
            if not hasattr(self, "recent_trade_zones"):
                self.recent_trade_zones = {}
            zone_info = self.recent_trade_zones.get(sym)
            if zone_info:
                z_elapsed = now_ts - zone_info.get("ts", 0)
                z_center = zone_info.get("price", 0.0)
                cur_entry = float(sig["entry_price"])
                z_dist_pips = abs(cur_entry - z_center) / pip if pip > 0 else 0.0
                # Jika berada di zona harga yang sama (Gold: <25 pip / Forex: <8 pip) dan belum lewat cooldown
                min_zone_pips = 25.0 if ("XAU" in sym or "GOLD" in sym) else 8.0
                max_zone_wait = 120.0 if getattr(self, "last_closed_was_profit_by_symbol", {}).get(sym, False) else 600.0
                if z_dist_pips < min_zone_pips and z_elapsed < max_zone_wait:
                    if sym == self.symbol:
                        self.last_diag = f"📍 SCALPING MEMPUNYAI ZONA ({sym}): Entry ditahan! Zona ({z_center:.2f} ± {min_zone_pips:.0f}p) baru saja dieksekusi ({int(z_elapsed)}s lalu). Dilarang spam order di zona yang sama."
                    continue

            # 7. PRE-TRADE CONFIRMATION GATE (Analisis Terlebih Dahulu Sebelum Open)
            # a. Verifikasi Proyeksi Walk-Forward
            fc_dir = "NETRAL"
            fc_prob = 50
            try:
                from forecast import forecast
                if len(sym_df) >= 60:
                    pip_s = get_symbol_pip_size(sym)
                    fc = forecast(sym_df, h=5, pip=pip_s, timeframe=self.timeframe, symbol=sym)
                    if fc and fc.get("ok"):
                        fc_dir = fc.get("direction", "NETRAL")
                        fc_prob = round(max(fc.get("prob_up", 0.5), fc.get("prob_down", 0.5)) * 100)
                        if sig["action"] == "BUY" and fc_dir == "TURUN" and fc_prob >= 58:
                            if sym == self.symbol:
                                self.last_diag = f"PRE-TRADE GATE ({sym}): BUY ditahan (Proyeksi 5-bar: TURUN {fc_prob}%)"
                            continue
                        if sig["action"] == "SELL" and fc_dir == "NAIK" and fc_prob >= 58:
                            if sym == self.symbol:
                                self.last_diag = f"PRE-TRADE GATE ({sym}): SELL ditahan (Proyeksi 5-bar: NAIK {fc_prob}%)"
                            continue
            except Exception as e:
                logger.debug(f"Pengecekan konfluensi forecast ({sym}) dilewati: {e}")

            # b. Verifikasi Deterministik Pra-Eksekusi (Spread & Dynamic Risk Validation)
            pip_size = get_symbol_pip_size(sym)
            spread_pips = (sym_ask - sym_bid) / pip_size if pip_size > 0 else 1.0

            # Filter Maksimum Spread: Batasi pelebaran spread agar tidak memakan margin (Gold max 4.0p, FX max 2.5p)
            max_allowed_spread = 4.0 if ("XAU" in sym or "GOLD" in sym) else 2.5
            if spread_pips > max_allowed_spread:
                if sym == self.symbol:
                    self.last_diag = f"SPREAD FILTER ({sym}): Spread melebar ({spread_pips:.1f}p > {max_allowed_spread:.1f}p). Order {sig['action']} ditahan."
                self.order_manager.log(f"⚠️ SPREAD FILTER [{sym}]: Order ditahan karena spread tinggi ({spread_pips:.1f}p > {max_allowed_spread:.1f}p)", "WARNING")
                continue

            self.order_manager.log(f"✅ PRE-TRADE GATE [{sym}]: Lolos verifikasi deterministik (Spread: {spread_pips:.1f}p | Setup: {sig.get('setup_score', 0)}%)", "INFO")

            # 8. Eksekusi Order Scalping Mandiri
            side_int = 1 if sig["action"] == "BUY" else -1

            active_tech = sig.get("technique") or getattr(self.params, "active_technique", "ALL")
            pos = self.order_manager.execute_scalping(
                symbol=sym,
                side=side_int,
                sl=sig["sl_price"],
                tp=sig["tp_price"],
                current_price=sig["entry_price"],
                spread_pips=spread_pips,
                lot=sig["lot"],
                target_usd=sig["target_usd"],
                technique=active_tech,
            )
            if pos:
                try:
                    feats = self._features(sym_df, side_int, sig["sl_price"], sig["tp_price"], sig["entry_price"], spread_pips)
                    self.pending_features[pos.ticket] = feats
                except Exception as ef:
                    logger.debug(f"Gagal simpan fitur latihan learner: {ef}")
                if not hasattr(self, "last_scalp_by_symbol"):
                    self.last_scalp_by_symbol = {}
                self.last_scalp_by_symbol[sym] = now_ts
                self.last_scalp_ts = now_ts

                # PILAR 2: Catat Zona Harga Eksekusi Terakhir (Anti-Spamming / One-Zone One-Trade)
                if not hasattr(self, "recent_trade_zones"):
                    self.recent_trade_zones = {}
                self.recent_trade_zones[sym] = {
                    "price": float(sig["entry_price"]),
                    "ts": now_ts,
                    "side": sig["action"],
                }
                all_positions.append(pos)
                tech_meta = AVAILABLE_TECHNIQUES.get(active_tech, {})
                diag_msg = f"🚀 MULTI-PAIR ENTRY: [{tech_meta.get('icon', '🎯')} {tech_meta.get('short_name', active_tech)}] Scalping {sig['action']} dieksekusi di {sym} @ {sig['entry_price']} | Target TP: +${sig['target_usd']:.2f} (SL: -${sig['risk_usd']:.2f})"
                if sym == self.symbol:
                    self.last_diag = diag_msg
                self.order_manager.log(diag_msg, "INFO")

    def _run_loop(self) -> None:
        """Loop eksekusi utama."""
        if len(self.df_candles) == 0:
            self.load_initial_candles()

        tick_counter = 0
        reconnect_counter = 0
        MT5_RECONNECT_INTERVAL = 30  # coba reconnect setiap 30 detik jika terputus

        while not self._stop_event.is_set():
            try:
                # 1. Update harga real-time
                if isinstance(self.executor, MT5Executor):
                    if getattr(self, "symbol_supported", True):
                        tick_ok = False
                        try:
                            with self.executor._lock:
                                tick = self.executor.mt5.symbol_info_tick(self.symbol)
                                term_ok = self.executor.mt5.terminal_info() is not None
                            if tick and tick.bid > 0:
                                self.current_bid = tick.bid
                                self.current_ask = tick.ask
                                self.executor.update_prices(self.symbol, tick.bid, tick.ask)
                                tick_ok = True
                                reconnect_counter = 0
                            elif term_ok:
                                tick_ok = True
                        except Exception as e:
                            logger.debug(f"Gagal update tick MT5: {e}")

                        # Auto-reconnect jika koneksi MT5 benar-benar terputus
                        if not tick_ok:
                            reconnect_counter += 1
                            if reconnect_counter >= MT5_RECONNECT_INTERVAL:
                                reconnect_counter = 0
                                term_alive = False
                                try:
                                    with self.executor._lock:
                                        term_alive = (self.executor.mt5.terminal_info() is not None)
                                except Exception:
                                    term_alive = False
                                if not term_alive:
                                    self._try_reconnect_mt5()
                    else:
                        self._simulate_tick()

                elif isinstance(self.executor, PaperExecutor):
                    self._simulate_tick()

                # 2. Cek Trailing Stop, Break-Even & Evaluasi by Data Pasar M1
                self.order_manager.manage_active_positions()

                # Pantau riwayat transaksi tertutup untuk aktivasi jeda anti-loss per simbol
                # 2. Cek apakah ada posisi tertutup baru untuk pembaruan status & proteksi loss beruntun
                hist = self.executor.get_trade_history()
                if hist:
                    if not hasattr(self, "_processed_closed_tickets"):
                        self._processed_closed_tickets = set(t.get("ticket") for t in hist[:50] if t.get("ticket"))

                    # Periksa trade baru yang tertutup (hist[0] adalah trade terbaru)
                    newly_closed = [t for t in hist[:15] if t.get("ticket") and t.get("ticket") not in self._processed_closed_tickets]
                    for ct in reversed(newly_closed):
                        c_ticket = ct.get("ticket")
                        self._processed_closed_tickets.add(c_ticket)
                        close_now = time.time()
                        self.last_closed_trade_ts = close_now
                        sym_closed = str(ct.get("symbol", self.symbol)).upper()
                        if not hasattr(self, "last_closed_trade_ts_by_symbol"):
                            self.last_closed_trade_ts_by_symbol = {}
                        if not hasattr(self, "last_closed_was_profit_by_symbol"):
                            self.last_closed_was_profit_by_symbol = {}
                        self.last_closed_trade_ts_by_symbol[sym_closed] = close_now

                        pnl = float(ct.get("profit", 0) or ct.get("pnl", 0))
                        reason = str(ct.get("reason", "")).upper()
                        is_profit = pnl >= 0.0
                        self.last_closed_was_profit_by_symbol[sym_closed] = is_profit

                        if not hasattr(self, "last_loss_by_symbol"):
                            self.last_loss_by_symbol = {}

                        # Hanya catat sebagai LOSS jika benar-benar rugi (pnl < -0.1). Posisi BE / Trailing SL yang cuan BUKAN kerugian!
                        if (not is_profit) and (pnl < -0.1) and ("STOP_LOSS" in reason or "SL" in reason or pnl < -1.0):
                            closed_side = str(ct.get("side", "")).upper()
                            self.last_loss_side = closed_side
                            self.last_loss_ts = close_now
                            # Ambil data Swing High / Low sebelum breakdown untuk proteksi struktur baru
                            swing_h_val = 0.0
                            swing_l_val = 0.0
                            try:
                                _, _, sym_df_curr = self._get_symbol_market_data(sym_closed)
                                if sym_df_curr is not None and len(sym_df_curr) >= 20:
                                    swing_h_val = float(sym_df_curr["high"].iloc[-20:].max())
                                    swing_l_val = float(sym_df_curr["low"].iloc[-20:].min())
                            except Exception:
                                pass

                            self.last_loss_by_symbol[sym_closed] = {
                                "side": closed_side,
                                "ts": close_now,
                                "pnl": pnl,
                                "price": float(ct.get("close_price", ct.get("close", ct.get("exit", 0.0))) or 0.0),
                                "swing_high": swing_h_val,
                                "swing_low": swing_l_val,
                            }
                            self.order_manager.log(
                                f"🛡️ PILAR 1 AKTIF [{sym_closed}]: Order #{c_ticket} kena SL (${pnl:.2f}). STOP LOSS TEMPAT MERUBAH VIEW! View {closed_side} di-invalidasi. Cooldown 10m & menunggu struktur baru.",
                                "WARNING",
                            )
                        elif is_profit:
                            # Jika ditutup cuan (TP, Trailing Runner, atau Anti-Balik Minus), hapus blokir invalidasi
                            self.last_loss_by_symbol.pop(sym_closed, None)

                        # Pencatatan Jurnal Otomatis (3 Pilar Trader Profesional: Stop Loss Tempat Merubah View, Scalping Mempunyai Zona, Jurnaling)
                        try:
                            from journal import get_trade_journal
                            jnl = get_trade_journal()
                            jnl.record_trade(
                                ticket=c_ticket,
                                symbol=sym_closed,
                                side=str(ct.get("side", "")).upper(),
                                open_time=str(ct.get("open_time", "")),
                                close_time=datetime.fromtimestamp(close_now).strftime("%Y-%m-%d %H:%M:%S"),
                                open_price=float(ct.get("open_price", ct.get("open", ct.get("entry_price", 0.0))) or 0.0),
                                close_price=float(ct.get("close_price", ct.get("close", ct.get("exit", 0.0))) or 0.0),
                                pnl=pnl,
                                reason=reason,
                                technique="EMA_TREND",
                                entry_in_zone=True,
                                zone_description="Dynamic Pocket EMA 8/21 Golden Zone",
                            )
                        except Exception as e_jnl:
                            logger.debug(f"Pencatatan jurnal dilewati: {e_jnl}")

                        # Continuous Real-Time Machine Learning: Perbarui bobot model AI segera setelah order tertutup
                        f = self.pending_features.pop(c_ticket, None)
                        if f is not None:
                            is_explore = c_ticket in self.pending_explore
                            self.pending_explore.discard(c_ticket)
                            lr = self.learner_explore if is_explore else self.learner
                            is_win = (pnl > 0)
                            lr.update(f, is_win)
                            self.order_manager.log(
                                f"🧠 AI BELAJAR REAL-TIME: Trade #{c_ticket} ({'WIN' if is_win else 'LOSS'} ${pnl:+.2f}) -> Bobot model diperbarui! Total data latih: {lr.n} sampel",
                                "INFO",
                            )

                # 2b. Cek Entri Scalping Otomatis (jika mode scalping aktif)
                self._try_scalping_entry()

                # 3. Sinkronkan candle atau deteksi penutupan bar
                tick_counter += 1
                if isinstance(self.executor, MT5Executor) and getattr(self, "symbol_supported", True):
                    # Sinkronkan candle dari MT5 setiap 2 detik
                    if tick_counter >= 2:
                        tick_counter = 0
                        try:
                            with self.executor._lock:
                                mt5 = self.executor.mt5
                                tf_map = {
                                    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                                    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                                    "D1": mt5.TIMEFRAME_D1, "W1": getattr(mt5, "TIMEFRAME_W1", mt5.TIMEFRAME_D1),
                                    "MN": getattr(mt5, "TIMEFRAME_MN1", mt5.TIMEFRAME_D1),
                                }
                                tf_val = tf_map.get(self.timeframe, mt5.TIMEFRAME_M5)
                                rates = mt5.copy_rates_from_pos(self.symbol, tf_val, 0, 3000)
                            if rates is not None and len(rates) > 0:
                                df = pd.DataFrame(rates)
                                df["time"] = pd.to_datetime(df["time"], unit="s")
                                new_df = df.set_index("time").sort_index()
                                prev_last = self.df_candles.index[-1] if len(self.df_candles) > 0 else None
                                self.df_candles = new_df
                                if prev_last is not None and new_df.index[-1] != prev_last:
                                    self._on_new_candle_closed(new_df.iloc[-2], new_df.iloc[:-1])
                        except Exception as e:
                            logger.debug(f"Error sinkron candle MT5: {e}")
                elif isinstance(self.executor, PaperExecutor):
                    if tick_counter >= 15: # Setiap 15 detik bar baru simulasi
                        tick_counter = 0
                        now = pd.Timestamp.now(tz="UTC")
                        spread = self.params.spread_pips * self.params.pip_size
                        new_row = pd.DataFrame([{
                            "open": self.current_bid,
                            "high": self.current_bid + spread * 1.5,
                            "low": self.current_bid - spread * 0.5,
                            "close": self.current_bid + spread * 0.2,
                            "tick_volume": 100,
                            "spread": int(self.params.spread_pips * 10),
                            "real_volume": 0
                        }], index=[now])
                        self.df_candles = pd.concat([self.df_candles, new_row]).tail(3000)
                        self._on_new_candle_closed(self.df_candles.iloc[-1], self.df_candles)

                # 4. Broadcast telemetri ke UI
                acc = self.executor.get_account_info()
                raw_positions = self.executor.get_positions()
                if hasattr(self, "order_manager") and hasattr(self.order_manager, "_peak_pnl_by_ticket"):
                    for p in raw_positions:
                        pk = self.order_manager._peak_pnl_by_ticket.get(p.ticket, float(getattr(p, "peak_pnl", 0.0) or 0.0))
                        pk = max(pk, float(p.floating_pnl))
                        self.order_manager._peak_pnl_by_ticket[p.ticket] = pk
                        p.peak_pnl = round(pk, 2)
                positions = [p.to_dict() for p in raw_positions]
                trades_hist = self.executor.get_trade_history()
                active_sig = self._compute_active_signal(self.df_candles)

                # Perbarui diagnosis real-time jika belum ada pesan khusus
                if positions:
                    self.last_diag = f"Posisi aktif ({len(positions)} order running): Memantau target cuan / trailing profit."
                elif not self.is_running:
                    self.last_diag = f"Bot SIAGA (PAUSE) - Tekan 'Start Bot' untuk eksekusi otomatis. Evaluasi setup: {active_sig.get('reason', 'Menunggu konfirmasi setup...')}"
                elif self.is_paused:
                    self.last_diag = f"Bot JEDA (PAUSED). Evaluasi setup: {active_sig.get('reason', 'Menunggu konfirmasi setup...')}"
                elif active_sig and active_sig.get("reason"):
                    self.last_diag = active_sig.get("reason")

                gemini_info = {"active": False, "status": "STANDBY", "reason": "Menunggu API Key"}
                try:
                    from gemini_analyst import get_gemini_analyst
                    analyst = get_gemini_analyst()
                    gemini_info = {
                        "active": analyst.is_available(),
                        "status": "ACTIVE" if analyst.is_available() else "STANDBY",
                        "last_analysis": analyst.last_analysis,
                    }
                except Exception:
                    pass

                self.broadcast("telemetry", {
                    "account": acc,
                    "mode": acc.get("mode", self.get_mode()),
                    "positions": positions,
                    "open_positions": positions,
                    "trade_history": trades_hist,
                    "symbol": self.symbol,
                    "symbol_supported": getattr(self, "symbol_supported", True),
                    "timeframe": self.timeframe,
                    "bid": round(self.current_bid, 5),
                    "ask": round(self.current_ask, 5),
                    "spread_pips": round((self.current_ask - self.current_bid) / self.params.pip_size, 2) if self.params.pip_size else 0.8,
                    "is_running": self.is_running,
                    "is_paused": self.is_paused,
                    "daily_r": self.order_manager.daily_r_pnl,
                    "cooldown": self.order_manager.cooldown_remaining,
                    "logs": self.order_manager.logs[-20:],
                    "diagnosis": self.last_diag,
                    "learner": self.learner.stats(),
                    "explore": self.get_explore_status(),
                    "active_signal": active_sig,
                    "technique_stats": self.get_technique_statistics(),
                    "active_technique": "EMA_TREND",
                    "active_technique_meta": AVAILABLE_TECHNIQUES.get("EMA_TREND", {}),
                    "trading_mode": getattr(self.params, "trading_mode", "scalping"),
                    "target_profit_usd": getattr(self.params, "target_profit_usd", 5.0),
                    "scalping_lot": getattr(self.params, "scalping_lot", 0.1),
                    "scalping_direction": getattr(self.params, "scalping_direction", "BOTH"),
                    "market_session": get_current_market_session(),
                    "session_filter_mode": getattr(self.params, "session_filter_mode", "STRICT"),
                    "gemini": gemini_info,
                    "max_positions_mode": getattr(self.params, "scalping_max_open_mode", "AUTO"),
                    "max_total_open": int(getattr(self.params, "scalping_max_open", 5)),
                    "multi_symbol": {
                        "enabled": bool(getattr(self, "multi_symbol_enabled", False)),
                        "watchlist": getattr(self, "watchlist", ["XAUUSD", "EURUSD", "GBPUSD", "US100"]),
                        "max_total_open": int(getattr(self.params, "scalping_max_open", 5)),
                        "max_open_per_symbol": int(getattr(self.params, "scalping_max_per_symbol", 5)),
                        "max_positions_mode": getattr(self.params, "scalping_max_open_mode", "AUTO"),
                        "open_count": len(positions),
                    },
                })

            except Exception as e:
                logger.error(f"Error di loop trading: {e}", exc_info=True)

            time.sleep(1.0)

    def _try_reconnect_mt5(self) -> None:
        """Coba reconnect ke MT5 secara otomatis menggunakan akun aktif tersimpan."""
        login_str = ""
        password = ""
        server = ""
        acc_type = "demo"
        term_path = None

        try:
            from account_vault import get_account_vault
            vault = get_account_vault()
            active_acc = vault.get_active_account()
            if active_acc and active_acc.get("account_type", "").lower() != "paper":
                cred = vault.get_account_credentials(active_acc.get("id"))
                if cred and cred.get("login"):
                    login_str = str(cred.get("login"))
                    password = cred.get("password") or cred.get("investor_password") or ""
                    server = str(cred.get("server") or "")
                    acc_type = str(cred.get("account_type", "demo"))
                    term_path = os.environ.get("MT5_PATH") or None
        except Exception:
            pass

        if not login_str or not login_str.isdigit():
            load_env()
            login_str = os.environ.get("MT5_LOGIN", "")
            password = os.environ.get("MT5_PASSWORD", "")
            server = os.environ.get("MT5_SERVER", "FinexBisnisSolusi-Demo")
            acc_type = os.environ.get("MT5_ACCOUNT_TYPE", "demo")
            term_path = os.environ.get("MT5_PATH") or None

        if not login_str or not login_str.isdigit():
            return

        logger.info(f"[AUTO-RECONNECT] Mencoba menghubungkan kembali ke MT5 #{login_str}...")
        self.order_manager.log(f"🔄 Auto-reconnect ke MT5 #{login_str}...", "INFO")
        try:
            self.switch_to_mt5(int(login_str), password, server, acc_type, term_path)
            self.order_manager.log(f"✅ Reconnect berhasil! Terhubung ke MT5 #{login_str}", "INFO")
        except Exception as e:
            logger.warning(f"[AUTO-RECONNECT] Gagal: {e}")
            self.order_manager.log(f"⚠️ Reconnect gagal: {e}. Akan coba lagi dalam 30 detik.", "WARNING")


    def get_status_summary(self) -> dict:
        if isinstance(self.executor, MT5Executor) and getattr(self, "symbol_supported", True):
            try:
                with self.executor._lock:
                    tick = self.executor.mt5.symbol_info_tick(self.symbol)
                if tick and tick.bid > 0:
                    self.current_bid = tick.bid
                    self.current_ask = tick.ask
                    self.executor.update_prices(self.symbol, tick.bid, tick.ask)
            except Exception:
                pass
        xau_bid = self.current_bid
        xau_ask = self.current_ask
        if isinstance(self.executor, MT5Executor):
            try:
                with self.executor._lock:
                    xtick = self.executor.mt5.symbol_info_tick("XAUUSD")
                    if xtick and xtick.bid > 0:
                        xau_bid = xtick.bid
                        xau_ask = xtick.ask
            except Exception:
                pass

        acc = self.executor.get_account_info()
        # Jika MT5 terputus, coba reconnect sekali dari sini (non-blocking)
        if isinstance(self.executor, MT5Executor) and not acc.get("connected", True):
            acc["mode"] = "MT5_RECONNECTING"
        raw_positions = self.executor.get_positions()
        if hasattr(self, "order_manager") and hasattr(self.order_manager, "_peak_pnl_by_ticket"):
            for p in raw_positions:
                pk = self.order_manager._peak_pnl_by_ticket.get(p.ticket, float(getattr(p, "peak_pnl", 0.0) or 0.0))
                pk = max(pk, float(p.floating_pnl))
                self.order_manager._peak_pnl_by_ticket[p.ticket] = pk
                p.peak_pnl = round(pk, 2)
        positions = [p.to_dict() for p in raw_positions]
        trades_hist = self.executor.get_trade_history()
        sig = self._compute_active_signal(self.df_candles)
        active_tech = "EMA_TREND"
        tech_meta = AVAILABLE_TECHNIQUES.get("EMA_TREND", {})
        tech_stats = self.get_technique_statistics()
        gemini_info = {"active": False, "status": "STANDBY", "reason": "Menunggu API Key"}
        gemini_news = None
        try:
            from gemini_analyst import get_gemini_analyst
            analyst = get_gemini_analyst()
            gemini_news = analyst.get_xauusd_news_sentiment(force_refresh=False)
            gemini_info = {
                "active": analyst.is_available(),
                "status": "ACTIVE" if analyst.is_available() else "STANDBY",
                "last_analysis": analyst.last_analysis,
                "news": gemini_news,
            }
        except Exception:
            pass

        if self.last_diag in ("", "Menunggu candle pertama..."):
            if positions:
                self.last_diag = f"Posisi aktif ({len(positions)} order running): Memantau target cuan / trailing profit."
            elif not self.is_running:
                self.last_diag = f"Bot SIAGA (PAUSE) - Tekan 'Start Bot' untuk eksekusi otomatis. Evaluasi setup: {sig.get('reason', 'Menunggu konfirmasi setup...')}"
            elif self.is_paused:
                self.last_diag = f"Bot JEDA (PAUSED). Evaluasi setup: {sig.get('reason', 'Menunggu konfirmasi setup...')}"
            elif sig and sig.get("reason"):
                self.last_diag = sig.get("reason")

        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "mode": acc.get("mode", self.get_mode()),
            "symbol": self.symbol,
            "symbol_supported": getattr(self, "symbol_supported", True),
            "timeframe": self.timeframe,
            "bid": self.current_bid,
            "ask": self.current_ask,
            "xau_bid": xau_bid,
            "xau_ask": xau_ask,
            "spread_pips": round((self.current_ask - self.current_bid) / self.params.pip_size, 2) if self.params.pip_size else 0.8,
            "account": acc,
            "positions": positions,
            "open_positions": positions,
            "trade_history": trades_hist,
            "daily_r": self.order_manager.daily_r_pnl,
            "cooldown": self.order_manager.cooldown_remaining,
            "logs": self.order_manager.logs[-50:],
            "diagnosis": self.last_diag,
            "learner": self.learner.stats(),
            "explore": self.get_explore_status(),
            "active_signal": sig,
            "trading_mode": getattr(self.params, "trading_mode", "scalping"),
            "target_profit_usd": getattr(self.params, "target_profit_usd", 5.0),
            "scalping_lot": getattr(self.params, "scalping_lot", 0.1),
            "scalping_direction": getattr(self.params, "scalping_direction", "BOTH"),
            "active_technique": active_tech,
            "active_technique_meta": tech_meta,
            "technique_stats": tech_stats,
            "market_session": get_current_market_session(),
            "session_filter_mode": getattr(self.params, "session_filter_mode", "STRICT"),
            "gemini": gemini_info,
            "gemini_news": gemini_news,
            "max_positions_mode": getattr(self.params, "scalping_max_open_mode", "AUTO"),
            "max_total_open": int(getattr(self.params, "scalping_max_open", 5)),
            "multi_symbol": {
                "enabled": bool(getattr(self, "multi_symbol_enabled", False)),
                "watchlist": getattr(self, "watchlist", ["XAUUSD", "EURUSD", "GBPUSD", "US100"]),
                "max_total_open": int(getattr(self.params, "scalping_max_open", 5)),
                "max_open_per_symbol": int(getattr(self.params, "scalping_max_per_symbol", 5)),
                "max_positions_mode": getattr(self.params, "scalping_max_open_mode", "AUTO"),
                "open_count": len(positions),
                "open_by_symbol": getattr(self, "open_by_symbol", {}),
            },
        }

    def set_target_profit(
        self,
        target_profit_usd: Optional[float] = None,
        trading_mode: Optional[str] = None,
        lot: Optional[float] = None,
        target_loss_usd: Optional[float] = None,
        direction: Optional[str] = None,
    ) -> None:
        """Ubah target profit USD ($2, $5, $10), mode trading, dan arah trading (DUAL/BUY/SELL) saat runtime."""
        overrides: dict = {}
        if target_profit_usd is not None and target_profit_usd > 0:
            tp_val = float(target_profit_usd)
            overrides["target_profit_usd"] = tp_val
            # Hitung batas rugi secara proporsional (RRR aman 1:1.20 - 1:1.25)
            if target_loss_usd is not None and target_loss_usd > 0:
                overrides["target_loss_usd"] = float(target_loss_usd)
            else:
                overrides["target_loss_usd"] = round(tp_val * 1.25, 2)
        else:
            tp_val = float(getattr(self.params, "target_profit_usd", 5.0) or 5.0)
            overrides["target_profit_usd"] = tp_val
            if target_loss_usd is not None and target_loss_usd > 0:
                overrides["target_loss_usd"] = float(target_loss_usd)
            else:
                overrides["target_loss_usd"] = round(tp_val * 1.20, 2)

        if trading_mode is not None:
            overrides["trading_mode"] = str(trading_mode)
        else:
            overrides["trading_mode"] = str(getattr(self.params, "trading_mode", "scalping") or "scalping")

        if lot is not None and lot > 0:
            overrides["scalping_lot"] = float(lot)
        else:
            overrides["scalping_lot"] = float(getattr(self.params, "scalping_lot", 0.1) or 0.1)

        if direction is not None:
            overrides["scalping_direction"] = str(direction).upper()
        else:
            overrides["scalping_direction"] = str(getattr(self.params, "scalping_direction", "BOTH") or "BOTH")

        self.param_overrides.update(overrides)
        self.params = build_params(self.symbol, self.timeframe, self.param_overrides)
        self.order_manager.params = self.params
        self._compute_active_signal()
        self.order_manager.log(
            f"KONFIGURASI BOT: Mode {self.params.trading_mode.upper()} | Arah: {self.params.scalping_direction} | Target Cuan: ${self.params.target_profit_usd:.2f} | Lot: {self.params.scalping_lot:.2f} Lot",
            "INFO",
        )
