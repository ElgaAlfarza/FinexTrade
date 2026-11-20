"""Tahap 4: Full-Stack Web Server & API (FastAPI + WebSockets).

Menyediakan dashboard interaktif untuk monitoring trading bot,
eksekusi manual, backtest studio langsung dari web, dan visualisasi chart.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backtest import compute_stats, run_backtest, with_params
from config import PRESETS, SYMBOL_SPECS, StrategyParams, build_params, symbol_decimals
from execution import (
    MT5Executor,
    TRADE_RECORDS_CSV,
    TRADE_RECORDS_ARCHIVE_CSV,
    get_symbol_pip_size,
    sync_trade_records_to_file,
    set_session_reset,
    clear_session_reset,
    get_session_reset_info,
)
from live_engine import LiveBotEngine
from strategy import align_htf, generate_signals, htf_structure, resample_ohlc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finex.server")

SESSION_TOKEN = os.environ.get("FINEX_AUTH_TOKEN") or secrets.token_urlsafe(32)

ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app = FastAPI(title="Finex Bot Trading Terminal", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Validasi Host header
    host_header = request.headers.get("host", "").split(":")[0]
    if host_header and host_header not in ("127.0.0.1", "localhost", "testserver"):
        return JSONResponse(status_code=403, content={"detail": "Forbidden: Host tidak diizinkan"})

    # Validasi Origin header jika ada
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS and not (origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:")):
        return JSONResponse(status_code=403, content={"detail": "Forbidden: Permintaan Cross-Origin ditolak"})

    path = request.url.path
    # Rute publik (bisa diakses tanpa token)
    if path in ("/", "/favicon.ico"):
        return await call_next(request)

    # Seluruh endpoint /api/ wajib menyertakan token sesi valid
    if path.startswith("/api/"):
        token = request.headers.get("X-Bot-Token") or request.query_params.get("token")
        if not token or not secrets.compare_digest(token, SESSION_TOKEN):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized: Sesi tidak valid atau token tidak disertakan"})

    return await call_next(request)

# Inisialisasi engine bot di background (Fokus default: XAUUSD Gold Spot)
engine = LiveBotEngine(symbol="XAUUSD", timeframe="M5", use_paper=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._send_locks: Dict[WebSocket, asyncio.Lock] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self._send_locks[websocket] = asyncio.Lock()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self._send_locks.pop(websocket, None)

    async def broadcast(self, message: dict):
        text = json.dumps(message)
        dead_conns = []
        for connection in list(self.active_connections):
            lock = self._send_locks.get(connection)
            if lock is None:
                continue
            # If connection is currently busy sending, skip this frame to prevent concurrent write collisions
            if lock.locked():
                continue
            try:
                async with lock:
                    await connection.send_text(text)
            except Exception:
                dead_conns.append(connection)
        for d in dead_conns:
            self.disconnect(d)

manager = ConnectionManager()
main_loop: Optional[asyncio.AbstractEventLoop] = None

def on_engine_telemetry(payload: dict):
    global main_loop
    if main_loop and main_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(manager.broadcast(payload), main_loop)
        except Exception:
            pass

engine.subscribe_telemetry(on_engine_telemetry)


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()

    # Auto-koneksi ke akun yang sedang aktif di brankas (AccountVault)
    active_acc = None
    try:
        from account_vault import get_account_vault
        vault = get_account_vault()
        active_acc = vault.get_active_account()
    except Exception as e:
        logger.warning(f"Gagal memuat active account dari vault: {e}")

    connected = False
    if active_acc:
        acc_type = active_acc.get("account_type", "demo").lower()
        if acc_type == "paper":
            engine.switch_to_paper()
            logger.info("Auto-connected ke Simulator Paper pada startup server!")
            connected = True
        elif active_acc.get("login"):
            try:
                login_val = int(active_acc.get("login"))
                pwd_val = active_acc.get("password") or active_acc.get("investor_password") or ""
                srv_val = active_acc.get("server")
                term_path = os.environ.get("MT5_PATH") or None
                engine.switch_to_mt5(login_val, pwd_val, srv_val, acc_type, term_path)
                logger.info(f"Auto-connected ke MT5 #{login_val} ({srv_val}) pada startup server!")
                connected = True
            except Exception as e:
                logger.warning(f"Auto-connect akun vault #{active_acc.get('login')} gagal ({e}), coba fallback .env...")

    if not connected:
        # Fallback ke kredensial .env jika vault belum terhubung
        login_str = os.environ.get("MT5_LOGIN", "")
        if login_str and login_str.isdigit():
            try:
                pwd = os.environ.get("MT5_PASSWORD", "")
                srv = os.environ.get("MT5_SERVER", "FinexBisnisSolusi-Demo")
                acc_type = os.environ.get("MT5_ACCOUNT_TYPE", "demo")
                term_path = os.environ.get("MT5_PATH") or None
                engine.switch_to_mt5(int(login_str), pwd, srv, acc_type, term_path)
                logger.info(f"Auto-connected ke MT5 #{login_str} dari .env pada startup server!")
            except Exception as e:
                logger.warning(f"Auto-connect MT5 gagal ({e}), fallback ke Paper.")
                engine.load_initial_candles()
        else:
            engine.load_initial_candles()

    # Jalankan bot otomatis saat server aktif
    engine.start()
    logger.info("Bot otomatis aktif & memantau peluang secara mandiri.")

    logger.info("Finex Bot Web Server aktif di http://localhost:8000")
    asyncio.create_task(telemetry_loop())
    asyncio.create_task(mt5_watchdog_loop())


async def telemetry_loop():
    """Streaming telemetri kontinu setiap 1 detik ke seluruh browser yang terhubung."""
    while True:
        try:
            if manager.active_connections:
                payload = {
                    "type": "telemetry",
                    "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                    "data": engine.get_status_summary(),
                }
                await manager.broadcast(payload)
        except Exception as e:
            logger.debug(f"Telemetry loop error: {e}")
        await asyncio.sleep(1.0)


async def mt5_watchdog_loop():
    """Watchdog: periksa koneksi MT5 setiap 30 detik, reconnect otomatis jika terputus."""
    from execution import MT5Executor
    await asyncio.sleep(15)  # Tunggu server stabil dulu
    while True:
        try:
            if isinstance(engine.executor, MT5Executor):
                acc = engine.executor.get_account_info()
                if not acc.get("connected", True):
                    logger.warning("[WATCHDOG] MT5 terputus! Mencoba reconnect otomatis...")
                    await asyncio.get_event_loop().run_in_executor(
                        None, engine._try_reconnect_mt5
                    )
        except Exception as e:
            logger.debug(f"[WATCHDOG] Error: {e}")
        await asyncio.sleep(30)


class LoginRequest(BaseModel):
    login: int
    password: str
    server: str = "FinexBisnisSolusi-Demo"
    account_type: str = "demo"  # "demo" atau "real"
    terminal_path: Optional[str] = None


class SwitchAccountRequest(BaseModel):
    account_id: str


@app.get("/api/status")
def get_status():
    return engine.get_status_summary()


@app.post("/api/account/login")
def account_login(req: LoginRequest):
    try:
        acc = engine.switch_to_mt5(
            login=req.login,
            password=req.password,
            server=req.server,
            account_type=req.account_type,
            terminal_path=req.terminal_path,
        )
        engine.broadcast("telemetry", engine.get_status_summary())
        return {"success": True, "account": acc}
    except Exception as e:
        logger.error(f"Gagal login MT5: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/account/disconnect")
def account_disconnect():
    acc = engine.switch_to_paper()
    engine.broadcast("telemetry", engine.get_status_summary())
    return {"success": True, "message": "Beralih ke Paper Simulator", "account": acc}


@app.post("/api/account/reconnect")
def account_reconnect():
    """Reconnect manual ke MT5 menggunakan kredensial tersimpan."""
    try:
        engine._try_reconnect_mt5()
        engine.broadcast("telemetry", engine.get_status_summary())
        acc = engine.executor.get_account_info()
        return {
            "success": acc.get("connected", False),
            "mode": acc.get("mode", "UNKNOWN"),
            "message": "Reconnect berhasil!" if acc.get("connected") else "Reconnect gagal, coba lagi atau login manual."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/account/current")
def get_current_account():
    return engine.executor.get_account_info()


@app.get("/api/accounts/list")
def list_saved_accounts():
    """Mengembalikan daftar akun tersimpan TANPA PASSWORD (Aman dari peretasan frontend)."""
    try:
        from account_vault import get_account_vault
        vault = get_account_vault()
        return {
            "success": True,
            "active_id": vault.get_active_account_id(),
            "accounts": vault.get_safe_accounts_list()
        }
    except Exception as e:
        logger.error(f"Gagal memuat daftar akun: {e}")
        return {"success": False, "error": str(e), "accounts": []}


@app.post("/api/accounts/switch")
def switch_saved_account(req: SwitchAccountRequest):
    """Beralih ke akun tersimpan secara 1-klik menggunakan ID internal secara aman."""
    try:
        from account_vault import get_account_vault
        vault = get_account_vault()
        acc_id = req.account_id
        cred = vault.get_account_credentials(acc_id)
        if not cred:
            raise HTTPException(status_code=404, detail=f"Akun '{acc_id}' tidak ditemukan di brankas.")

        acc_type = cred.get("account_type", "demo").lower()
        if acc_type == "paper":
            acc_info = engine.switch_to_paper()
            vault.set_active_account(acc_id)
            engine.broadcast("telemetry", engine.get_status_summary())
            return {
                "success": True,
                "account_id": acc_id,
                "message": "Berhasil beralih ke Simulator Paper!",
                "account": acc_info
            }

        login = int(cred.get("login"))
        pwd = cred.get("password") or cred.get("investor_password") or ""
        srv = cred.get("server")

        acc_info = engine.switch_to_mt5(
            login=login,
            password=pwd,
            server=srv,
            account_type=acc_type,
        )
        vault.set_active_account(acc_id)
        engine.broadcast("telemetry", engine.get_status_summary())
        return {
            "success": True,
            "account_id": acc_id,
            "message": f"Berhasil beralih ke {cred.get('name')} (#{login})!",
            "account": acc_info
        }
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Gagal beralih akun ke {req.account_id if hasattr(req, 'account_id') else req}: {err_msg}")
        if "-10005" in err_msg or "IPC timeout" in err_msg:
            err_msg = (
                f"Aplikasi MT5 belum mengenali server '{srv}'. "
                "Buka MT5 di desktop -> Berkas (File) -> Buka Rekening (Open an Account) -> "
                "Ketik 'Markets4you' lalu Enter, kemudian klik Batal. Setelah itu coba klik lagi di web!"
            )
        raise HTTPException(status_code=400, detail=err_msg)


@app.get("/api/history")
def get_history():
    return {"trades": engine.executor.get_trade_history()}


@app.get("/api/history/export")
def export_history_file():
    trades = engine.executor.get_trade_history()
    sync_trade_records_to_file(trades)
    target_file = TRADE_RECORDS_CSV
    if not os.path.exists(target_file) or os.path.getsize(target_file) <= 120:
        if os.path.exists(TRADE_RECORDS_ARCHIVE_CSV):
            target_file = TRADE_RECORDS_ARCHIVE_CSV
    if os.path.exists(target_file):
        return FileResponse(
            target_file,
            media_type="text/csv",
            filename=f"Riwayat_Semua_Trade_Finex_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
        )
    raise HTTPException(status_code=404, detail="File riwayat transaksi belum tersedia")


@app.post("/api/account/reset-session")
def reset_account_session():
    """Reset riwayat tampilan transaksi dan profit hari/bulan ini ke 0.
    Semua dataset latih dan bobot AI tetap utuh."""
    acc = engine.executor.get_account_info()
    cur_bal = float(acc.get("balance", 10.0) or 10.0)
    mt5_inst = getattr(engine.executor, "mt5", None)
    reset_info = set_session_reset(initial_balance=cur_bal, mt5_instance=mt5_inst)
    # Clear executor internal caches
    if hasattr(engine.executor, "_hist_cache"):
        engine.executor._hist_cache = []
        engine.executor._hist_cache_ts = 0.0
    if hasattr(engine.executor, "_deals_summary_cache"):
        engine.executor._deals_summary_cache = None
        engine.executor._deals_cache_ts = 0.0
    if hasattr(engine.executor, "closed_trades"):
        engine.executor.closed_trades = [t for t in engine.executor.closed_trades if t.get("close_epoch", 0) >= reset_info["reset_time"]]
    if hasattr(engine, "order_manager"):
        engine.order_manager.daily_r_pnl = 0.0
        engine.order_manager.risk_usd_by_ticket.clear()
    logger.info(f"[RESET SESI] Sesi baru diaktifkan per {reset_info['reset_time_str']} dengan saldo awal ${cur_bal:.2f}. Dataset latih tetap utuh.")
    return {"ok": True, "reset_info": reset_info, "balance": cur_bal}



@app.post("/api/bot/start")
def start_bot():
    engine.start()
    return {"success": True, "message": "Bot dimulai"}


@app.post("/api/bot/pause")
def pause_bot():
    engine.pause()
    return {"success": True, "message": "Bot di-pause"}


@app.post("/api/bot/resume")
def resume_bot():
    engine.resume()
    return {"success": True, "message": "Bot di-resume"}


@app.post("/api/bot/stop")
def stop_bot():
    engine.stop()
    return {"success": True, "message": "Bot dihentikan"}


@app.post("/api/bot/kill-switch")
def kill_switch():
    closed = engine.emergency_kill_switch()
    return {"success": True, "closed_positions": closed, "message": f"{closed} posisi ditutup paksa"}


@app.post("/api/trades/break-even/{ticket}")
def set_break_even(ticket: int):
    positions = [p for p in engine.executor.get_positions() if p.ticket == ticket]
    if not positions:
        raise HTTPException(status_code=404, detail="Posisi tidak ditemukan")
    p = positions[0]
    pip = get_symbol_pip_size(p.symbol)
    digits = symbol_decimals(p.symbol)
    be_sl = round(
        p.open_price + engine.params.be_buffer_pips * pip if p.side == "BUY" else p.open_price - engine.params.be_buffer_pips * pip,
        digits,
    )
    success = engine.executor.modify_sl_tp(ticket, be_sl, p.tp)
    if success:
        p.be_active = True
        p.sl = be_sl
        engine.order_manager.log(f"Manual BE Order #{ticket} ({p.symbol}) -> {be_sl:.{digits}f}", "INFO")
    return {"success": success, "new_sl": be_sl}


@app.post("/api/trades/close/{ticket}")
def close_trade(ticket: int):
    success = engine.executor.close_trade(ticket, reason="MANUAL_UI")
    if success:
        engine.order_manager.log(f"Manual Close Order #{ticket} dari UI", "INFO")
        engine.broadcast("telemetry", engine.get_status_summary())
        return {"success": True, "message": f"Order #{ticket} berhasil ditutup!"}

    # Cek detail kenapa gagal
    err_msg = getattr(engine.executor, "last_order_error", "")
    if isinstance(engine.executor, MT5Executor):
        try:
            t_info = engine.executor.mt5.terminal_info()
            if t_info and not t_info.trade_allowed:
                err_msg = "Tombol 'Algo Trading' di aplikasi MetaTrader 5 sedang nonaktif (merah). Silakan klik tombol 'Algo Trading' di MT5 agar berwarna HIJAU (atau tekan tombol Ctrl+E di keyboard MT5), atau tutup langsung lewat tanda silang (X) di MT5."
        except Exception:
            pass
    raise HTTPException(status_code=400, detail=err_msg or f"Gagal menutup order #{ticket} di MT5")


class QuickOrderRequest(BaseModel):
    symbol: Optional[str] = "XAUUSD"
    side: str = "BUY"  # "BUY" atau "SELL"
    lots: float = 0.01


_last_quick_order_time = 0.0

@app.post("/api/trades/quick-order")
def quick_order(req: QuickOrderRequest):
    global _last_quick_order_time
    now = time.time()
    if now - _last_quick_order_time < 1.0:
        raise HTTPException(status_code=429, detail="Terlalu banyak permintaan: Harap tunggu 1 detik antar order manual.")
    _last_quick_order_time = now

    sym = (req.symbol or "XAUUSD").upper().strip()
    side = req.side.upper().strip()

    if side not in ("BUY", "SELL"):
        raise HTTPException(
            status_code=400,
            detail=f"Sinyal saat ini masih [{side}]. Order hanya dapat dieksekusi jika sinyal BUY atau SELL."
        )

    sym_params = build_params(sym, engine.timeframe)
    pip = sym_params.pip_size
    decimals = symbol_decimals(sym)

    cur_price = 0.0
    if isinstance(engine.executor, MT5Executor):
        try:
            with engine.executor._lock:
                if not engine.executor.mt5.symbol_select(sym, True):
                    raise HTTPException(status_code=400, detail=f"Simbol {sym} tidak tersedia di broker Finex MT5.")
                tick = engine.executor.mt5.symbol_info_tick(sym)
            if tick:
                cur_price = tick.ask if side == "BUY" else tick.bid
        except HTTPException:
            raise
        except Exception:
            pass
    if cur_price <= 0 and sym == engine.symbol:
        cur_price = engine.current_ask if side == "BUY" else engine.current_bid
    if cur_price <= 0:
        cur_price = float(sym_params.default_price)

    sl_pips_default = max(20.0, sym_params.min_sl_pips * 1.5)
    sl_dist = sl_pips_default * pip
    tp_dist = sl_pips_default * 1.5 * pip
    sl = round(cur_price - sl_dist if side == "BUY" else cur_price + sl_dist, decimals)
    tp = round(cur_price + tp_dist if side == "BUY" else cur_price - tp_dist, decimals)

    # Validasi dan adaptasi lot otomatis berdasarkan margin bebas akun
    acc = engine.executor.get_account_info()
    free_m = float(acc.get("free_margin", 0.0) or 0.0)
    order_lot = max(0.01, float(req.lots or 0.01))

    if hasattr(engine.executor, "get_required_margin"):
        specs = engine.executor.get_symbol_trade_specs(sym, pip, cur_price)
        vmin = specs.get("min_lot", 0.01)
        req_m = engine.executor.get_required_margin(sym, side, order_lot, cur_price)
        min_m = engine.executor.get_required_margin(sym, side, vmin, cur_price)

        if free_m > 0 and free_m < min_m:
            raise HTTPException(
                status_code=400,
                detail=f"Margin bebas akun (${free_m:.2f}) tidak cukup untuk membuka {sym} (butuh margin minimal ${min_m:.2f} untuk {vmin} lot). Silakan tutup posisi yang sedang berjalan terlebih dahulu atau tambah deposit."
            )
        elif free_m > 0 and req_m > free_m * 0.90:
            engine.order_manager.log(f"💡 ADAPTASI MODAL: Lot disesuaikan otomatis dari {order_lot} ke {vmin} lot sesuai sisa margin (${free_m:.2f}).", "INFO")
            order_lot = vmin

    pos = engine.executor.open_trade(
        symbol=sym,
        side=side,
        lots=order_lot,
        sl=sl,
        tp=tp,
        risk_pips=sl_pips_default,
        comment=f"Manual-{side}"
    )
    if pos:
        engine.order_manager.log(f"MANUAL ORDER #{pos.ticket}: {side} {pos.lots} lot {sym} @ {cur_price:.{decimals}f} | SL: {sl:.{decimals}f} | TP: {tp:.{decimals}f}", "SIGNAL")
        engine.broadcast("telemetry", engine.get_status_summary())
        return {"success": True, "position": pos.to_dict()}
    else:
        err_msg = getattr(engine.executor, 'last_order_error', None) or "Gagal membuka order di broker/MT5"
        raise HTTPException(status_code=400, detail=err_msg)


@app.post("/api/trades/close-all")
def close_all_trades():
    closed = engine.executor.close_all(reason="MANUAL_CLOSE_ALL")
    engine.order_manager.log(f"JUAL SEMUA: {closed} posisi berhasil ditutup sekaligus dari UI", "INFO")
    engine.broadcast("telemetry", engine.get_status_summary())
    return {
        "success": True,
        "closed_positions": closed,
        "message": f"{closed} posisi berhasil ditutup!" if closed > 0 else "Tidak ada posisi terbuka untuk ditutup."
    }


class ConfigureMarketRequest(BaseModel):
    symbol: str = "EURUSD"
    timeframe: str = "M5"


@app.post("/api/bot/configure")
def configure_market(req: ConfigureMarketRequest):
    engine.set_market(req.symbol, req.timeframe)
    engine.broadcast("telemetry", engine.get_status_summary())
    return {
        "success": True,
        "symbol": engine.symbol,
        "timeframe": engine.timeframe,
        "message": f"Market diubah ke {engine.symbol} ({engine.timeframe})"
    }


class TargetProfitRequest(BaseModel):
    target_profit_usd: Optional[float] = 5.0
    trading_mode: Optional[str] = "scalping"
    scalping_lot: Optional[float] = None
    target_loss_usd: Optional[float] = None
    scalping_direction: Optional[str] = None


@app.post("/api/bot/target-profit")
def set_target_profit(req: TargetProfitRequest):
    engine.set_target_profit(
        target_profit_usd=req.target_profit_usd,
        trading_mode=req.trading_mode,
        lot=req.scalping_lot,
        target_loss_usd=req.target_loss_usd,
        direction=req.scalping_direction,
    )
    engine.broadcast("telemetry", engine.get_status_summary())
    return {
        "success": True,
        "target_profit_usd": getattr(engine.params, "target_profit_usd", 5.0),
        "trading_mode": getattr(engine.params, "trading_mode", "scalping"),
        "scalping_lot": getattr(engine.params, "scalping_lot", 0.02),
        "scalping_direction": getattr(engine.params, "scalping_direction", "BOTH"),
        "message": f"Target profit diatur ke ${getattr(engine.params, 'target_profit_usd', 5.0):.2f} ({getattr(engine.params, 'trading_mode', 'scalping')}) | Lot: {getattr(engine.params, 'scalping_lot', 0.02):.2f} | Arah: {getattr(engine.params, 'scalping_direction', 'BOTH')}"
    }


class DirectionRequest(BaseModel):
    direction: str = "BOTH"  # "BOTH", "BUY_ONLY", "SELL_ONLY"


@app.post("/api/bot/direction")
def set_bot_direction(req: DirectionRequest):
    engine.set_target_profit(direction=req.direction)
    engine.broadcast("telemetry", engine.get_status_summary())
    return {
        "success": True,
        "direction": getattr(engine.params, "scalping_direction", "BOTH"),
        "message": f"Arah trading bot diatur ke: {getattr(engine.params, 'scalping_direction', 'BOTH')}"
    }


class LotRequest(BaseModel):
    lot: float = 0.1


@app.post("/api/bot/lot")
def set_bot_lot(req: LotRequest):
    if req.lot <= 0:
        raise HTTPException(status_code=400, detail="Ukuran lot harus > 0")
    engine.set_target_profit(lot=req.lot)
    engine.broadcast("telemetry", engine.get_status_summary())
    return {
        "success": True,
        "scalping_lot": getattr(engine.params, "scalping_lot", 0.1),
        "message": f"Ukuran lot bot diatur ke: {getattr(engine.params, 'scalping_lot', 0.1):.2f} Lot"
    }


class MarketModeRequest(BaseModel):
    enabled: bool = False  # False = 1 Market (Layar aktif), True = Semua Market (Multi-Pair)


@app.post("/api/bot/market-mode")
def set_market_mode(req: MarketModeRequest):
    res = engine.set_multi_symbol_mode(req.enabled)
    engine.broadcast("telemetry", engine.get_status_summary())
    return res


class TechniqueRequest(BaseModel):
    technique: str = "EMA_TREND"


@app.post("/api/bot/technique")
def set_bot_technique(req: TechniqueRequest):
    res = engine.set_active_technique(req.technique)
    engine.broadcast("telemetry", engine.get_status_summary())
    return res


@app.get("/api/bot/technique-stats")
def get_technique_stats():
    return {
        "active_technique": "EMA_TREND",
        "stats": engine.get_technique_statistics(),
    }


class SessionFilterRequest(BaseModel):
    mode: str = "STRICT"  # "STRICT" (hanya London/NY) atau "ALL" (semua sesi diizinkan)


@app.post("/api/bot/session-filter")
def set_session_filter(req: SessionFilterRequest):
    res = engine.set_session_filter_mode(req.mode)
    engine.broadcast("telemetry", engine.get_status_summary())
    return res


class MaxPositionsRequest(BaseModel):
    mode: str = "AUTO"  # "1", "2", "3", "4", "5", atau "AUTO"


@app.post("/api/bot/max-positions")
def set_max_positions(req: MaxPositionsRequest):
    res = engine.set_max_positions_mode(req.mode)
    engine.broadcast("telemetry", engine.get_status_summary())
    return res


class GeminiConfigRequest(BaseModel):
    api_key: str


@app.post("/api/bot/gemini-config")
def set_gemini_config(req: GeminiConfigRequest):
    from gemini_analyst import get_gemini_analyst
    analyst = get_gemini_analyst()
    ok = analyst.set_api_key(req.api_key)
    if ok:
        engine.order_manager.log("🤖 AI GEMINI ANALYST: API Key berhasil diaktifkan untuk analisis Pre-Trade!", "INFO")
        engine.broadcast("telemetry", engine.get_status_summary())
        return {"success": True, "message": "API Key Gemini tersimpan aman & Analis Pre-Trade AKTIF!"}
    raise HTTPException(status_code=400, detail="API Key tidak boleh kosong")


@app.get("/api/bot/gemini-status")
def get_gemini_status():
    from gemini_analyst import get_gemini_analyst
    analyst = get_gemini_analyst()
    return {
        "active": analyst.is_available(),
        "status": "ACTIVE" if analyst.is_available() else "STANDBY",
        "last_analysis": analyst.last_analysis,
    }


@app.get("/api/gemini/news")
def get_gemini_news(force_refresh: bool = False):
    """Ambil analisis berita dan sentimen makroekonomi terkini seputar pergerakan XAUUSD dari AI Gemini."""
    from gemini_analyst import get_gemini_analyst
    analyst = get_gemini_analyst()
    return analyst.get_xauusd_news_sentiment(force_refresh=force_refresh)


@app.post("/api/gemini/news/refresh")
def refresh_gemini_news():
    """Segarkan berita & sentimen makroekonomi XAUUSD secara instan."""
    from gemini_analyst import get_gemini_analyst
    analyst = get_gemini_analyst()
    res = analyst.get_xauusd_news_sentiment(force_refresh=True)
    engine.broadcast("telemetry", engine.get_status_summary())
    return res


class ExploreRequest(BaseModel):
    enabled: bool = True
    every_bars: int = 1
    max_open: int = 3
    daily_loss_r: float = 10.0


@app.get("/api/bot/explore")
def get_explore():
    return engine.get_explore_status()


@app.post("/api/bot/explore")
def set_explore(req: ExploreRequest):
    try:
        engine.set_explore(req.enabled, req.every_bars, req.max_open, req.daily_loss_r)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    engine.broadcast("telemetry", engine.get_status_summary())
    return {"success": True, **engine.get_explore_status()}


_forecast_cache: dict = {}


def _load_fallback_ohlc(sym: str, tf: str, min_bars: int = 1200) -> pd.DataFrame:
    """Muat data historis lokal yang sesuai simbol & timeframe (tanpa mencampur simbol lain)."""
    if engine.symbol == sym and engine.timeframe == tf and len(engine.df_candles) >= min(200, min_bars):
        return engine.df_candles.copy()
    csv_path = f"data/{sym}_{tf}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time").sort_index().tail(5000)
    from generate_sample_data import generate_market_data
    df = generate_market_data(sym, tf, bars=max(min_bars, 800))
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


@app.get("/api/forecast")
def get_forecast(symbol: Optional[str] = None, timeframe: Optional[str] = None, horizon: int = 5):
    """Proyeksi probabilistik h candle ke depan sesuai simbol & timeframe yang dipilih user."""
    from forecast import forecast

    sym = (symbol or engine.symbol).upper()
    tf = (timeframe or engine.timeframe).upper()
    h = max(1, min(int(horizon), 20))
    sym_params = build_params(sym, tf)

    df = None
    broker_supported = False
    is_live_mt5 = isinstance(engine.executor, MT5Executor)

    try:
        if is_live_mt5:
            with engine.executor._lock:
                mt5 = engine.executor.mt5
                if mt5.symbol_select(sym, True):
                    broker_supported = True
                    tf_map = {
                        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                        "D1": mt5.TIMEFRAME_D1, "W1": getattr(mt5, "TIMEFRAME_W1", mt5.TIMEFRAME_D1),
                        "MN": getattr(mt5, "TIMEFRAME_MN1", mt5.TIMEFRAME_D1),
                    }
                    tf_val = tf_map.get(tf, mt5.TIMEFRAME_M5)
                    r = mt5.copy_rates_from_pos(sym, tf_val, 0, 3000)
                    if r is not None and len(r) > 400:
                        raw_df = pd.DataFrame(r)
                        raw_df["time"] = pd.to_datetime(raw_df["time"], unit="s")
                        df = raw_df.set_index("time").sort_index().iloc[:-1]  # 100% candle tertutup (tanpa lookahead)

        if df is None:
            df = _load_fallback_ohlc(sym, tf, min_bars=1200)
            if len(df) > 400:
                df = df.iloc[:-1]

        if df is None or len(df) < 350:
            return {"ok": False, "symbol": sym, "timeframe": tf, "reason": f"Data historis {sym} ({tf}) belum cukup untuk kalibrasi walk-forward"}

        sig = engine.active_signal if (engine and hasattr(engine, "active_signal")) else {}
        bias_action = sig.get("action") if (sym == engine.symbol and sig.get("action") in ("BUY", "SELL")) else None
        cache_key = (sym, tf, h, str(df.index[-1]), len(df), broker_supported, bias_action)
        if cache_key in _forecast_cache:
            return _forecast_cache[cache_key]

        out = forecast(df, h=h, pip=sym_params.pip_size, timeframe=tf, symbol=sym, bias_action=bias_action)
        out["broker_supported"] = broker_supported if is_live_mt5 else True
        if out.get("ok"):
            if is_live_mt5 and not broker_supported:
                out["note"] = f"Simbol {sym} tidak tersedia di server Finex MT5 — proyeksi ini memakai data simulasi lokal."
            elif not is_live_mt5:
                out["note"] = "Mode Paper/CSV: dikalibrasi dari data historis lokal, bukan feed live MT5."

        if len(_forecast_cache) > 64:
            _forecast_cache.clear()
        _forecast_cache[cache_key] = out
        return out
    except Exception as e:
        logger.warning(f"Forecast error ({sym} {tf}): {e}")
        return {"ok": False, "symbol": sym, "timeframe": tf, "reason": str(e)}


@app.get("/api/gemini/chart-analysis")
def get_gemini_chart_analysis(symbol: Optional[str] = None, timeframe: Optional[str] = None, force_refresh: bool = False):
    """Analisis struktur chart & level visual harga masa depan oleh AI Gemini."""
    from gemini_analyst import get_gemini_analyst
    analyst = get_gemini_analyst()

    sym = (symbol or engine.symbol).upper()
    tf = (timeframe or engine.timeframe).upper()

    df = None
    cur_price = 0.0
    is_live_mt5 = isinstance(engine.executor, MT5Executor)

    try:
        if is_live_mt5:
            with engine.executor._lock:
                mt5 = engine.executor.mt5
                if mt5.symbol_select(sym, True):
                    tick = mt5.symbol_info_tick(sym)
                    if tick:
                        cur_price = float(tick.bid)
                    tf_map = {
                        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                        "D1": mt5.TIMEFRAME_D1, "W1": getattr(mt5, "TIMEFRAME_W1", mt5.TIMEFRAME_D1),
                        "MN": getattr(mt5, "TIMEFRAME_MN1", mt5.TIMEFRAME_D1),
                    }
                    tf_val = tf_map.get(tf, mt5.TIMEFRAME_M5)
                    r = mt5.copy_rates_from_pos(sym, tf_val, 0, 80)
                    if r is not None and len(r) > 10:
                        raw_df = pd.DataFrame(r)
                        raw_df["time"] = pd.to_datetime(raw_df["time"], unit="s")
                        df = raw_df.set_index("time").sort_index()

        if df is None:
            df = _load_fallback_ohlc(sym, tf, min_bars=100)

        if df is not None and len(df) > 0 and cur_price <= 0:
            cur_price = float(df["close"].iloc[-1])

        result = analyst.analyze_chart_structure(
            symbol=sym,
            timeframe=tf,
            df_candles=df,
            current_price=cur_price,
            force_refresh=force_refresh,
        )
        return result
    except Exception as e:
        logger.warning(f"Gemini chart analysis error ({sym} {tf}): {e}")
        return {
            "bias": "KONSOLIDASI_RANGE",
            "confidence": 70,
            "outlook_title": "ANALISIS MASA DEPAN GRAFIK",
            "future_summary": f"Pemindaian visual sedang memproses {sym} ({tf}).",
            "predicted_path": "Memantau struktur candle...",
            "trader_action": "Amati reaksi harga di sekitar level pivot.",
            "key_levels": [],
            "gemini_active": analyst.is_available(),
            "symbol": sym,
            "timeframe": tf,
        }


@app.get("/api/candles")
def get_candles(symbol: Optional[str] = None, timeframe: Optional[str] = None, count: int = 100):
    sym = (symbol or engine.symbol).upper()
    tf = (timeframe or engine.timeframe).upper()
    digits = symbol_decimals(sym)
    is_live_mt5 = isinstance(engine.executor, MT5Executor)
    broker_supported = not is_live_mt5

    df = pd.DataFrame()
    if is_live_mt5:
        try:
            with engine.executor._lock:
                mt5 = engine.executor.mt5
                if mt5.symbol_select(sym, True):
                    broker_supported = True
                    tf_map = {
                        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                        "D1": mt5.TIMEFRAME_D1, "W1": getattr(mt5, "TIMEFRAME_W1", mt5.TIMEFRAME_D1),
                        "MN": getattr(mt5, "TIMEFRAME_MN1", mt5.TIMEFRAME_D1),
                    }
                    tf_val = tf_map.get(tf, mt5.TIMEFRAME_M5)
                    # Ambil jumlah bar optimal agar responsif real-time dan kalkulasi struktur akurat
                    fetch_n = min(max(int(count) + 80, 250), 500)
                    rates = mt5.copy_rates_from_pos(sym, tf_val, 0, fetch_n)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df["time"] = pd.to_datetime(df["time"], unit="s")
                        df = df.set_index("time").sort_index()
        except Exception as e:
            logger.warning(f"Error copy_rates ({sym} {tf}): {e}")

    if df.empty:
        df = _load_fallback_ohlc(sym, tf, min_bars=max(int(count) + 200, 800))

    if df.empty:
        return {"candles": [], "fib_levels": None, "ma": None, "symbol": sym, "timeframe": tf, "broker_supported": broker_supported, "decimals": digits}

    df_display = df.tail(int(count))

    # Gunakan parameter spesifik simbol + timeframe yang sedang dilihat
    overrides = engine.param_overrides if sym == engine.symbol else None
    p = build_params(sym, tf, overrides)
    htf_rule = p.htf_rule
    try:
        htf = resample_ohlc(df, htf_rule)
        info = align_htf(df, htf_structure(htf, p.pivot_k))
    except Exception:
        info = pd.DataFrame()

    candles_list = []
    for idx_time, row in df_display.iterrows():
        candles_list.append({
            "time": idx_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(idx_time, "strftime") else str(idx_time),
            "open": round(float(row["open"]), digits),
            "high": round(float(row["high"]), digits),
            "low": round(float(row["low"]), digits),
            "close": round(float(row["close"]), digits),
        })

    # Ambil info struktur terbaru
    last_info = info.iloc[-1] if not info.empty else None
    fib_data = None
    if last_info is not None and not pd.isna(last_info.get("trend")):
        trend = int(last_info["trend"])
        bL, bH = float(last_info["buy_L"]), float(last_info["buy_H"])
        sH, sL = float(last_info["sell_H"]), float(last_info["sell_L"])
        if trend == 1 and (bH - bL) > 0:
            fib_data = {
                "trend": "UP (BULLISH)",
                "leg_low": round(bL, digits),
                "leg_high": round(bH, digits),
                "fib_50": round(bH - 0.5 * (bH - bL), digits),
                "fib_618": round(bH - 0.618 * (bH - bL), digits),
                "fib_786": round(bH - 0.786 * (bH - bL), digits),
                "tp_target": round(bH + p.tp_extension * (bH - bL), digits),
            }
        elif trend == -1 and (sH - sL) > 0:
            fib_data = {
                "trend": "DOWN (BEARISH)",
                "leg_high": round(sH, digits),
                "leg_low": round(sL, digits),
                "fib_50": round(sL + 0.5 * (sH - sL), digits),
                "fib_618": round(sL + 0.618 * (sH - sL), digits),
                "fib_786": round(sL + 0.786 * (sH - sL), digits),
                "tp_target": round(sL - p.tp_extension * (sH - sL), digits),
            }

    ma_data = None
    if getattr(p, "use_ma_filter", False):
        try:
            from strategy import ma_direction, ma_htf_frame
            mf = ma_htf_frame(df, p)
            d = int(ma_direction(mf.tail(1), p.ma_mode)[0])
            tail = mf.tail(len(df_display))
            ma_data = {
                "type": p.ma_type, "fast": p.ma_fast, "slow": p.ma_slow, "mode": p.ma_mode,
                "direction": {1: "BUY_ONLY", -1: "SELL_ONLY", 0: "NETRAL"}[d],
                "fast_series": [None if pd.isna(v) else round(float(v), digits) for v in tail["ma_fast"]],
                "slow_series": [None if pd.isna(v) else round(float(v), digits) for v in tail["ma_slow"]],
            }
        except Exception as e:
            logger.warning(f"Error hitung MA overlay: {e}")

    return {
        "candles": candles_list,
        "fib_levels": fib_data,
        "ma": ma_data,
        "symbol": sym,
        "timeframe": tf,
        "broker_supported": broker_supported,
        "decimals": digits,
    }


class BacktestRequest(BaseModel):
    csv_file: str = "MT5_LIVE:EURUSD"
    timeframe: str = "M5"
    sl_mode: str = "structure"
    use_break_even: bool = False
    use_session_filter: bool = False
    risk_pct: float = 1.0
    min_rr: float = 1.5
    spread_pips: float = 0.8
    wick_ratio: float = 0.35
    use_ma_filter: bool = False
    ma_type: str = "EMA"
    ma_fast: int = 21
    ma_slow: int = 50
    ma_mode: str = "price_slow"


@app.post("/api/backtest/run")
def api_run_backtest(req: BacktestRequest):
    tf = (req.timeframe or "M5").upper()
    src = req.csv_file.strip()

    # Deteksi simbol dari nama sumber data
    sym = "EURUSD"
    src_up = src.upper()
    for candidate in SYMBOL_SPECS:
        if candidate in src_up:
            sym = candidate
            break

    df = pd.DataFrame()
    data_source_label = src

    if src_up.startswith("MT5_LIVE:"):
        sym = src_up.split(":", 1)[1].strip() or "EURUSD"
        if not isinstance(engine.executor, MT5Executor):
            raise HTTPException(status_code=400, detail="MT5 belum terhubung. Hubungkan akun MT5 Finex terlebih dahulu atau pilih dataset CSV.")
        with engine.executor._lock:
            mt5 = engine.executor.mt5
            if not mt5.symbol_select(sym, True):
                raise HTTPException(status_code=400, detail=f"Simbol {sym} tidak tersedia di server broker Finex MT5.")
            tf_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
            tf_val = tf_map.get(tf, mt5.TIMEFRAME_M5)
            rates = mt5.copy_rates_from_pos(sym, tf_val, 0, 5000)
        if rates is None or len(rates) < 100:
            raise HTTPException(status_code=400, detail=f"Gagal mengambil riwayat candle MT5 untuk {sym} ({tf}).")
        raw_df = pd.DataFrame(rates)
        raw_df["time"] = pd.to_datetime(raw_df["time"], unit="s")
        df = raw_df.set_index("time").sort_index().iloc[:-1]
        data_source_label = f"MT5 Live Finex ({sym} {tf}, {len(df)} bar tertutup)"
    else:
        if not os.path.exists(src):
            raise HTTPException(status_code=404, detail=f"File {src} tidak ditemukan di server")
        df = pd.read_csv(src)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        df = df[~df.index.duplicated(keep="first")]

    base_p = build_params(sym, tf)
    sl_mode = req.sl_mode if req.sl_mode in ("structure", "candle") else "structure"
    p = with_params(
        base_p,
        sl_mode=sl_mode,
        use_break_even=req.use_break_even,
        use_session_filter=req.use_session_filter,
        risk_pct=req.risk_pct,
        min_rr=req.min_rr,
        spread_pips=req.spread_pips,
        wick_ratio=req.wick_ratio,
        use_ma_filter=req.use_ma_filter,
        ma_type=req.ma_type.upper(),
        ma_fast=req.ma_fast,
        ma_slow=req.ma_slow,
        ma_mode=req.ma_mode,
    )

    trades = run_backtest(df, p)
    stats = compute_stats(trades, p)
    digits = symbol_decimals(sym)

    # Format daftar trade agar ramah JSON
    trade_list = []
    if not trades.empty:
        for _, t in trades.iterrows():
            trade_list.append({
                "entry_time": str(t["entry_time"]),
                "exit_time": str(t["exit_time"]),
                "side": t["side"],
                "entry": round(float(t["entry"]), digits),
                "sl": round(float(t["sl"]), digits),
                "tp": round(float(t["tp"]), digits),
                "exit": round(float(t["exit"]), digits),
                "reason": t["reason"],
                "r": round(float(t["r"]), 2),
                "risk_pips": round(float(t["risk_pips"]), 1),
                "bars": int(t["bars"]),
            })

    return {
        "source": data_source_label,
        "symbol": sym,
        "timeframe": tf,
        "bars": int(len(df)),
        "stats": {
            "trades": stats["trades"],
            "win_rate": round(stats.get("win_rate", 0) * 100, 1),
            "win_rate_ci_low": round(stats.get("win_rate_ci95", (0, 0))[0] * 100, 1) if not pd.isna(stats.get("win_rate_ci95", (0, 0))[0]) else 0.0,
            "win_rate_ci_high": round(stats.get("win_rate_ci95", (0, 0))[1] * 100, 1) if not pd.isna(stats.get("win_rate_ci95", (0, 0))[1]) else 0.0,
            "avg_win_r": round(stats.get("avg_win_r", 0), 2),
            "avg_loss_r": round(stats.get("avg_loss_r", 0), 2),
            "breakeven_win_rate": round(stats.get("breakeven_win_rate", 0) * 100, 1) if not pd.isna(stats.get("breakeven_win_rate", 0)) else 0.0,
            "profit_factor": round(stats.get("profit_factor", 0), 2) if stats.get("profit_factor") != float("inf") else 999.0,
            "expectancy_r": round(stats.get("expectancy_r", 0), 3),
            "total_r": round(stats.get("total_r", 0), 2),
            "max_drawdown": round(stats.get("max_drawdown", 0) * 100, 1),
            "end_equity": round(stats.get("end_equity", 10000), 2),
            "start_equity": round(stats.get("start_equity", 10000), 2),
            "max_consecutive_losses": stats.get("max_consecutive_losses", 0),
            "tp_count": stats.get("tp_count", 0),
            "sl_count": stats.get("sl_count", 0),
            "be_count": stats.get("be_count", 0),
            "timeout_count": stats.get("timeout_count", 0),
        },
        "equity_curve": stats.get("equity_curve", []),
        "trades": trade_list,
    }


@app.get("/api/datasets")
def list_datasets():
    files = []
    if isinstance(engine.executor, MT5Executor):
        for sym in ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US100"):
            files.append(f"MT5_LIVE:{sym}")
    data_dir = "data"
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".csv"):
                files.append(f"data/{f}")
    return {"datasets": files}


@app.get("/api/journal")
def get_trading_journal(limit: int = 100):
    """Jurnal Trading Otomatis (3 Pilar Trader Profesional: Stop Loss Tempat Merubah View, Scalping Mempunyai Zona, Jurnaling)."""
    from journal import get_trade_journal
    jnl = get_trade_journal()
    records_csv = os.path.join(os.path.dirname(__file__), "data", "trade_records.csv")
    if os.path.exists(records_csv):
        jnl.sync_from_trade_records(records_csv)
    return {
        "success": True,
        "summary": jnl.get_summary(),
        "entries": jnl.entries[:limit],
    }


@app.get("/api/datasets/xauusd")
def get_xauusd_datasets_info():
    """Informasi seluruh dataset multi-timeframe XAUUSD (2004 - 2026) dan laporan 90/10 split."""
    dataset_dir = os.path.join(os.path.dirname(__file__), "data", "historical_xauusd")
    items = []
    total_bytes = 0
    if os.path.exists(dataset_dir):
        for f in os.listdir(dataset_dir):
            if f.startswith("XAU_") and f.endswith(".csv"):
                fp = os.path.join(dataset_dir, f)
                sz = os.path.getsize(fp)
                total_bytes += sz
                tf = f.replace("XAU_", "").replace("_data.csv", "")
                items.append({
                    "filename": f,
                    "timeframe": tf,
                    "size_mb": round(sz / (1024 * 1024), 2),
                    "path": fp,
                })

    report = None
    report_path = os.path.join(os.path.dirname(__file__), "data", "train_test_evaluation_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as rf:
                report = json.load(rf)
        except Exception:
            pass

    return {
        "success": True,
        "count": len(items),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "datasets": sorted(items, key=lambda x: x["size_mb"], reverse=True),
        "learner": engine.learner.stats(),
        "evaluation_report": report,
    }


@app.post("/api/datasets/train")
def run_dataset_training(limit: int = 200000, train_ratio: float = 0.90):
    """Latih ulang model AI menggunakan 90% Data Latih & 10% Data Uji (Out-of-Sample)."""
    from learner import Learner
    try:
        from train_test_split_xauusd import run_train_test_split
        res = run_train_test_split(sample_size=limit, train_ratio=train_ratio)
        engine.learner = Learner(os.path.join(os.path.dirname(__file__), "learner_state.json"))
        sample_count = res.get("learner_stats", {}).get("samples", limit)
        engine.order_manager.log(f"🧠 MODEL AI DILATIH FORMAL (90% Latih / 10% Uji): {sample_count:,} trade terkalibrasi!", "INFO")
        engine.broadcast("telemetry", engine.get_status_summary())
        return {
            "success": True,
            "total_samples": sample_count,
            "win_rate": res.get("train_results", {}).get("win_rate", 39.2),
            "report": res,
        }
    except Exception as e:
        logger.error(f"Gagal latih dataset: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/model/eurusd")
async def get_eurusd_model_info():
    """Mengembalikan informasi arsitektur, parameter, dan metrik model EURUSD hasil latih notebook."""
    state_path = os.path.join(os.path.dirname(__file__), "data", "eurusd_model_state.json")
    if not os.path.exists(state_path):
        return {"success": False, "message": "Model EURUSD belum dilatih"}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"success": True, "model": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/news/calendar")
async def get_economic_calendar():
    """Mengembalikan kalender ekonomi real-time, jadwal CPI/NFP/GDP/FOMC, dan status News Shield."""
    try:
        from news_radar import get_news_radar
        radar = get_news_radar()
        summary = radar.get_summary()
        return {"success": True, "calendar": summary}
    except Exception as e:
        logger.error(f"Gagal memuat kalender berita: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/news/shield/toggle")
async def toggle_news_shield():
    """Mengaktifkan / menonaktifkan proteksi otomatis Auto-Pause saat rilis berita High-Impact."""
    try:
        from news_radar import get_news_radar
        radar = get_news_radar()
        radar.news_shield_enabled = not radar.news_shield_enabled
        status = radar.get_news_shield_status()
        return {
            "success": True,
            "shield_enabled": radar.news_shield_enabled,
            "message": f"News Shield {'Diaktifkan' if radar.news_shield_enabled else 'Dinonaktifkan'}",
            "status": status,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/news/cpi-prediction")
async def get_cpi_prediction(event: Optional[str] = "AUTO"):
    """Mengembalikan nowcasting prediksi PPI/CPI, leading factors, dan playbook XAUUSD."""
    try:
        from cpi_predictor import get_cpi_predictor
        pred = get_cpi_predictor()
        return pred.get_cpi_nowcast_summary(event or "AUTO")
    except Exception as e:
        logger.error(f"Gagal kalkulasi nowcasting Makro (PPI/CPI): {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/news/cpi-sniper/toggle")
async def toggle_cpi_sniper(enabled: Optional[bool] = None, mode: Optional[str] = None):
    """Mengubah status aktif dan mode Macro Big Move Sniper."""
    try:
        from cpi_predictor import get_cpi_predictor
        pred = get_cpi_predictor()
        if enabled is not None:
            pred.sniper_enabled = bool(enabled)
        else:
            pred.sniper_enabled = not pred.sniper_enabled
        if mode in ("PRE_POSITIONING", "POST_MOMENTUM"):
            pred.sniper_mode = mode
        pred.save_config()
        return {
            "success": True,
            "sniper_enabled": pred.sniper_enabled,
            "sniper_mode": pred.sniper_mode,
            "message": f"Macro Auto-Sniper {'Diaktifkan' if pred.sniper_enabled else 'Dinonaktifkan'} (Mode: {pred.sniper_mode})",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/news/cpi-sniper/execute-manual")
async def execute_manual_cpi_sniper(event: Optional[str] = "AUTO", symbol: Optional[str] = "XAUUSD"):
    """Eksekusi 1-Klik posisi antisipasi PPI/CPI (BUY atau SELL sesuai perhitungan by data + TP Cuan Besar)."""
    try:
        from cpi_predictor import get_cpi_predictor
        from execution import get_symbol_pip_size, symbol_decimals
        pred = get_cpi_predictor()
        target_sym = str(symbol or "XAUUSD").upper()
        dir_res = pred.predict_market_direction_by_data(event or "AUTO", target_sym)
        action = dir_res["action"] # "BUY" atau "SELL"
        active_ev = dir_res["event_name"]
        
        pip_sz = get_symbol_pip_size(target_sym)
        decimals = symbol_decimals(target_sym)
        
        lot = pred.target_lot
        is_gold = "XAU" in target_sym
        sl_pips = pred.sl_buffer_pips if is_gold else 25.0
        tp_pips = 100.0 if is_gold else 50.0
        
        if action == "BUY":
            cur_px = engine.current_ask if (target_sym == engine.symbol and engine.current_ask > 0) else (4000.0 if is_gold else 1.1000)
            entry_price = cur_px
            sl_price = round(entry_price - (sl_pips * pip_sz), decimals)
            tp_price = round(entry_price + (tp_pips * pip_sz), decimals)
            action_side = 1
        else:
            cur_px = engine.current_bid if (target_sym == engine.symbol and engine.current_bid > 0) else (4000.0 if is_gold else 1.1000)
            entry_price = cur_px
            sl_price = round(entry_price + (sl_pips * pip_sz), decimals)
            tp_price = round(entry_price - (tp_pips * pip_sz), decimals)
            action_side = -1
        
        pos = engine.order_manager.execute_scalping(
            symbol=target_sym,
            side=action_side,
            sl=sl_price,
            tp=tp_price,
            current_price=entry_price,
            spread_pips=2.5 if is_gold else 1.2,
            lot=lot,
            target_usd=pred.target_cuan_usd,
            technique=f"{active_ev}_SNIPER",
        )
        if pos:
            engine.order_manager.log(
                f"🎯 EKSEKUSI {active_ev} SNIPER ({action}): {action} {lot} lot {target_sym} @ {entry_price:.{decimals}f} | SL: {sl_price:.{decimals}f} | TP: {tp_price:.{decimals}f} (+${pred.target_cuan_usd:.2f}) | {dir_res['rationale']}",
                "INFO",
            )
            return {
                "success": True,
                "ticket": pos.ticket,
                "action": action,
                "message": f"Posisi {action} {target_sym} {active_ev} Sniper berhasil dibuka! Ticket #{pos.ticket} @ {entry_price:.{decimals}f}",
                "pos": pos.to_dict(),
            }
        else:
            return {"success": False, "message": "Gagal membuka posisi (kuota penuh atau margin tidak cukup)"}
    except Exception as e:
        return {"success": False, "error": str(e)}



@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    token = websocket.query_params.get("token") or websocket.headers.get("x-bot-token")
    if not token or not secrets.compare_digest(token, SESSION_TOKEN):
        await websocket.close(code=1008)  # 1008 Policy Violation
        return

    await manager.connect(websocket)
    try:
        # Kirim status awal
        await websocket.send_text(json.dumps({
            "type": "initial_status",
            "data": engine.get_status_summary(),
        }))
        while True:
            # Dengarkan perintah atau heartbeat dari client
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as e:
        logger.debug(f"WS telemetry client error/disconnect: {e}")
    finally:
        manager.disconnect(websocket)


# Sajikan file frontend Single Page Application (SPA)
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        auth_inject = f"""<script>
window.__BOT_AUTH_TOKEN__ = "{SESSION_TOKEN}";
(function() {{
    const origFetch = window.fetch;
    window.fetch = function(url, options) {{
        options = options || {{}};
        options.headers = options.headers || {{}};
        if (options.headers instanceof Headers) {{
            if (!options.headers.has('X-Bot-Token')) options.headers.set('X-Bot-Token', window.__BOT_AUTH_TOKEN__);
        }} else if (Array.isArray(options.headers)) {{
            options.headers.push(['X-Bot-Token', window.__BOT_AUTH_TOKEN__]);
        }} else {{
            options.headers['X-Bot-Token'] = window.__BOT_AUTH_TOKEN__;
        }}
        return origFetch(url, options);
    }};
}})();
</script>"""
        if "<head>" in content:
            content = content.replace("<head>", f"<head>\n{auth_inject}", 1)
        else:
            content = auth_inject + content

        resp = HTMLResponse(content=content)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
    return HTMLResponse("<h3>Dashboard frontend sedang disiapkan...</h3>")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
