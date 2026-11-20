"""Jembatan ke MetaTrader 5.

PENTING: paket Python `MetaTrader5` hanya berjalan di Windows dengan terminal MT5
terpasang. Modul ini belum dijalankan terhadap akun Finex (lingkungan pembuatnya
Linux). Uji pertama dilakukan di PC Windows Anda dengan `python export_history.py --check`.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

TIMEFRAMES = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1",
}


def load_env(path: str = ".env") -> None:
    """Baca file .env sederhana (KEY=VALUE). Kata sandi tidak pernah ditulis ke kode."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _import_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError(
            "Paket MetaTrader5 tidak tersedia. Jalankan di Windows dengan terminal MT5 terpasang, "
            "lalu: pip install MetaTrader5"
        ) from exc
    return mt5


class MT5Session:
    """Context manager koneksi. Menolak akun real kecuali require_demo=False."""

    def __init__(self, login: int, password: str, server: str,
                 terminal_path: str | None = None, require_demo: bool = True):
        self.login, self.password, self.server = login, password, server
        self.terminal_path, self.require_demo = terminal_path, require_demo
        self.mt5 = _import_mt5()

    def __enter__(self) -> "MT5Session":
        kwargs = dict(login=self.login, password=self.password, server=self.server)
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        if not self.mt5.initialize(**kwargs):
            err = self.mt5.last_error()
            if err and err[0] == -10003:
                raise RuntimeError("Aplikasi MetaTrader 5 belum dibuka atau belum terpasang di komputer Anda. Silakan buka aplikasi MT5 Finex terlebih dahulu di Windows, lalu coba lagi.")
            raise RuntimeError(f"Gagal terhubung ke MT5: {err}")
        info = self.mt5.account_info()
        if info is None:
            self.mt5.shutdown()
            raise RuntimeError(f"Login gagal, info akun kosong: {self.mt5.last_error()}")

        is_demo = (info.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO)
        allow_real = (os.environ.get("ALLOW_LIVE_REAL", "0") == "1")

        if not is_demo and not allow_real:
            self.mt5.shutdown()
            raise RuntimeError("Akun ini BUKAN akun demo (Akun REAL terdeteksi). Akses ditolak demi keamanan (butuh ALLOW_LIVE_REAL=1).")

        currency = str(getattr(info, "currency", "USD") or "USD").upper()
        if currency in ("CNT", "USDC"):
            self.mt5.shutdown()
            raise RuntimeError(f"Akun berdenominasi Cent ({currency}) belum didukung sampai normalisasi unit tervalidasi penuh.")

        if self.require_demo and not is_demo:
            self.mt5.shutdown()
            raise RuntimeError("Akun ini BUKAN akun demo. Koneksi dihentikan demi keamanan.")
        return self

    def __exit__(self, *exc) -> None:
        self.mt5.shutdown()

    def account_summary(self) -> dict:
        a = self.mt5.account_info()
        t = self.mt5.terminal_info()
        return dict(
            login=a.login, server=a.server, currency=a.currency,
            demo=a.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO,
            balance=a.balance, equity=a.equity, leverage=a.leverage,
            akun_boleh_trading=a.trade_allowed,
            akun_boleh_ea=a.trade_expert,
            terminal_algo_trading_aktif=t.trade_allowed,
        )

    def pip_value_per_lot(self, symbol: str, pip_size: float) -> float:
        self.mt5.symbol_select(symbol, True)
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Simbol {symbol} tidak ditemukan di server")
        tick = self.mt5.symbol_info_tick(symbol)
        ref_price = float(tick.ask or tick.bid) if tick and (tick.ask or tick.bid) else float(getattr(info, "bid", 0.0) or 1.0)
        if ref_price > 0 and pip_size > 0:
            calc = self.mt5.order_calc_profit(self.mt5.ORDER_TYPE_BUY, symbol, 1.0, ref_price, ref_price + float(pip_size))
            if calc is not None and calc > 0:
                return float(calc)
        from risk import estimate_pip_value
        return estimate_pip_value(symbol, pip_size)

    def export_rates(self, symbol: str, tf: str, date_from: datetime, date_to: datetime, out_csv: str) -> int:
        """Ekspor candle (harga BID) ke CSV. date_from/date_to sebaiknya berzona UTC."""
        if not self.mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Gagal memilih simbol {symbol}: {self.mt5.last_error()}")
        rates = self.mt5.copy_rates_range(symbol, getattr(self.mt5, TIMEFRAMES[tf]), date_from, date_to)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Tidak ada data untuk {symbol} {tf}: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")  # waktu server broker, bukan UTC
        df.to_csv(out_csv, index=False)
        return len(df)
