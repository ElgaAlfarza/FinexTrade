"""Parameter strategi dan risiko. Semua angka di sini adalah HIPOTESIS awal
yang harus diuji lewat backtest, bukan nilai yang sudah terbukti.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    # --- Struktur tren (timeframe lebih besar, dibentuk dari data entry) ---
    htf_rule: str = "30min"        # aturan resample timeframe tren
    pivot_k: int = 2               # jumlah candle di tiap sisi untuk konfirmasi swing

    # --- Fibonacci retracement (zona masuk) dan target ---
    fib_top: float = 0.5           # tepi dangkal zona (50%)
    fib_bottom: float = 0.786      # tepi dalam zona (78.6%)
    tp_extension: float = 0.27     # TP di extension -0.27 dari leg

    # --- Filter Moving Average (MA) pada timeframe tren (HTF) ---
    # CATATAN: nilai default di bawah adalah HIPOTESIS awal yang umum dipakai, BUKAN setting dari
    # video TikTok (setting video tidak tersedia). Wajib diuji ulang di data XAUUSD asli.
    use_ma_filter: bool = False    # aktifkan filter MA (searah tren HTF)
    ma_type: str = "EMA"           # "SMA" atau "EMA"
    ma_fast: int = 21              # periode MA cepat (0 = tidak dipakai)
    ma_slow: int = 50              # periode MA lambat
    ma_mode: str = "price_slow"    # "price_slow": buy jika close HTF > MA lambat (sell sebaliknya)
                                   # "cross": buy jika MA cepat > MA lambat (sell sebaliknya)
                                   # "price_both": buy jika close > MA cepat DAN MA lambat

    # --- MODE LATIHAN (khusus akun DEMO/PAPER): entry sesering mungkin supaya model belajar ---
    explore_mode: bool = False      # aktif/nonaktif
    explore_every_bars: int = 1     # entry latihan paling cepat setiap N candle tertutup (M1: 1 = tiap 1 menit)
    explore_max_open: int = 3       # batas posisi terbuka bersamaan selama mode latihan
    explore_daily_loss_r: float = 10.0  # circuit breaker harian saat mode latihan (lebih longgar dari 3R)
    explore_sl_atr: float = 1.5     # jarak SL = N x ATR(14) timeframe entry (dijepit ke batas min/max SL)
    explore_rr: float = 1.5         # TP = RR x jarak SL

    # --- Validasi entry (candle rejection di timeframe entry) ---
    wick_ratio: float = 0.35       # minimal porsi sumbu terhadap range candle

    # --- Stop loss dan rasio ---
    sl_mode: str = "structure"     # "structure" = di luar awal leg, "candle" = di luar candle rejection
    sl_buffer_pips: float = 2.0
    min_sl_pips: float = 5.0
    max_sl_pips: float = 40.0
    min_rr: float = 1.2

    # --- Proteksi & Trade Management (Trader Edge) ---
    use_break_even: bool = True    # pindahkan SL ke entry saat floating profit mencapai BE target
    break_even_r: float = 1.0      # level R untuk trigger Break-Even (default 1.0 R)
    be_buffer_pips: float = 0.5    # buffer pip di atas/bawah entry saat pindah ke BE (mengunci biaya komisi)

    # --- MODE TRADING & TARGET CUAN CEPAT ($2, $5, $10) ---
    trading_mode: str = "scalping" # "scalping" (aktif kejar target $2/$5/$10) atau "fibonacci" (swing santai)
    active_technique: str = "EMA_TREND"  # Fokus tunggal: MA EMA Trend Following (EMA 8 / 21 / 50 Ribbon)
    target_profit_usd: float = 5.0 # Target ambil untung otomatis per trade ($2, $5, $10)
    target_loss_usd: float = 5.0   # Batas pengaman rugi darurat per trade (dalam $ - Rasio 1:1)
    scalping_lot: float = 0.10     # Ukuran lot scalping konsisten 0.10 lot
    scalping_max_open: int = 5     # Maksimal total posisi terbuka bersamaan di seluruh portofolio
    scalping_max_open_mode: str = "AUTO" # Pilihan posisi: '1', '2', '3', '4', '5', atau 'AUTO'
    scalping_max_per_symbol: int = 5 # Maksimal posisi per simbol aset
    scalping_multi_symbol: bool = False # False: Bot fokus 1 market aktif
    scalping_interval_sec: int = 15 # Evaluasi sinyal scalping tiap N detik
    scalping_direction: str = "BOTH" # "BOTH", "BUY_ONLY", "SELL_ONLY"

    # --- Filter Sesi & Spread (Praktek Trader Nyata) ---
    max_spread_pips: float = 6.0   # tolak entry jika spread melebihi batas ini
    use_session_filter: bool = False # Default False agar bot bisa trading langsung
    session_filter_mode: str = "STRICT"
    session_start_utc: int = 7     # 07:00 UTC (London Open)
    session_end_utc: int = 20      # 20:00 UTC (NY Close)

    # --- Biaya eksekusi (dalam pip) ---
    pip_size: float = 0.0001
    spread_pips: float = 0.8
    slippage_pips: float = 0.2

    # --- Manajemen risiko & Batas Aman ---
    risk_pct: float = 1.0          # risiko per trade (% ekuitas)
    max_daily_loss_r: float = 3.0  # circuit breaker: berhenti trading jika rugi sebesar ini (dalam R)
    max_bars_in_trade: int = 500
    cooldown_bars: int = 3
    magic_number: int = 20261106   # identifier order bot di MT5

    # --- Identitas Aset & Kategori ---
    category: str = "COMMODITY"
    name: str = "Gold Spot / Emas"
    default_price: float = 3710.00


AVAILABLE_TECHNIQUES = {
    "EMA_TREND": {
        "id": "EMA_TREND",
        "name": "Trend Following Moving Average (EMA 8 / 21 / 50 Ribbon)",
        "short_name": "MA EMA Trend",
        "icon": "fa-chart-line",
        "badge_color": "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
        "desc": "Fokus Tunggal & Presisi Tinggi: Mengikuti arus tren besar (EMA 8 > 21 > 50 untuk BUY, EMA 8 < 21 < 50 untuk SELL), pullback retest dinamis di EMA 8/21, dan konfirmasi momentum candle.",
    },
}


# Preset awal per timeframe entry.
PRESETS = {
    "M1": dict(htf_rule="15min", max_sl_pips=25.0, min_sl_pips=3.0),
    "M5": dict(htf_rule="30min", max_sl_pips=40.0, min_sl_pips=5.0),
    "M15": dict(htf_rule="1h", max_sl_pips=50.0, min_sl_pips=8.0),
    "M30": dict(htf_rule="2h", max_sl_pips=70.0, min_sl_pips=10.0),
    "H1": dict(htf_rule="4h", max_sl_pips=100.0, min_sl_pips=15.0),
    "H4": dict(htf_rule="1D", max_sl_pips=200.0, min_sl_pips=30.0),
    "D1": dict(htf_rule="1W", max_sl_pips=500.0, min_sl_pips=50.0),
    "W1": dict(htf_rule="1M", max_sl_pips=1000.0, min_sl_pips=100.0),
    "MN": dict(htf_rule="3M", max_sl_pips=2000.0, min_sl_pips=200.0),
}

# Spesifikasi umum aset instrumen (Forex, Emas/Logam, Indeks, Kripto) — dikalibrasi pada ukuran M5
SYMBOL_SPECS = {
    # FOREX (Mata Uang)
    "EURUSD": dict(category="FOREX", name="Euro / US Dollar", pip_size=0.0001, spread_pips=0.8, max_spread_pips=3.0, min_sl_pips=5.0, max_sl_pips=40.0, default_price=1.11385),
    "GBPUSD": dict(category="FOREX", name="Poundsterling / US Dollar", pip_size=0.0001, spread_pips=1.2, max_spread_pips=4.5, min_sl_pips=6.0, max_sl_pips=50.0, default_price=1.29500),
    "USDJPY": dict(category="FOREX", name="US Dollar / Japanese Yen", pip_size=0.01, spread_pips=1.0, max_spread_pips=4.5, min_sl_pips=6.0, max_sl_pips=45.0, default_price=153.500),
    "AUDUSD": dict(category="FOREX", name="Australian Dollar / USD", pip_size=0.0001, spread_pips=1.0, max_spread_pips=3.5, min_sl_pips=5.0, max_sl_pips=40.0, default_price=0.65500),

    # EMAS & KOMODITAS (Logam Mulia & Energi)
    "XAUUSD": dict(use_ma_filter=True, category="COMMODITY", name="Gold Spot / Emas", pip_size=0.1, spread_pips=1.8, max_spread_pips=5.0, sl_buffer_pips=2.0, be_buffer_pips=3.0, min_sl_pips=5.0, max_sl_pips=60.0, default_price=3710.00),
    "XAGUSD": dict(category="COMMODITY", name="Silver Spot / Perak", pip_size=0.01, spread_pips=3.0, max_spread_pips=6.5, min_sl_pips=10.0, max_sl_pips=120.0, default_price=32.40),
    "XTIUSD": dict(category="COMMODITY", name="WTI Crude Oil / Minyak", pip_size=0.01, spread_pips=3.0, max_spread_pips=8.0, min_sl_pips=10.0, max_sl_pips=150.0, default_price=70.50),

    # INDEKS SAHAM AS (Tersedia Live di Broker Finex MT5)
    "US100": dict(use_ma_filter=True, category="INDEX", name="Nasdaq 100 Index", pip_size=1.0, spread_pips=2.0, max_spread_pips=6.5, sl_buffer_pips=3.0, be_buffer_pips=1.5, min_sl_pips=15.0, max_sl_pips=200.0, default_price=20500.0),

    # KRIPTO (Buka 24/7 Termasuk Akhir Pekan — Simulasi/Broker Non-BAPPEBTI)
    "BTCUSD": dict(use_session_filter=False, category="CRYPTO", name="Bitcoin / US Dollar", pip_size=1.0, spread_pips=15.0, max_spread_pips=50.0, min_sl_pips=100.0, max_sl_pips=2500.0, default_price=76500.0),
    "ETHUSD": dict(use_session_filter=False, category="CRYPTO", name="Ethereum / US Dollar", pip_size=0.1, spread_pips=2.5, max_spread_pips=8.0, min_sl_pips=20.0, max_sl_pips=350.0, default_price=2950.0),
}


def symbol_decimals(symbol: str) -> int:
    """Jumlah digit desimal harga wajar berdasarkan instrumen."""
    sym = symbol.upper()
    if any(k in sym for k in ("XAU", "BTC", "ETH", "US100", "US500", "US30", "JP225", "XTI")):
        return 2
    if "JPY" in sym or "XAG" in sym:
        return 3
    return 5


def build_params(symbol: str = "XAUUSD", timeframe: str = "M5", overrides: dict | None = None) -> StrategyParams:
    """Gabungkan spesifikasi simbol + skala timeframe tanpa menimpa batas SL aset non-Forex."""
    sym = (symbol or "XAUUSD").upper()
    tf = (timeframe or "M5").upper()
    sym_kw = dict(SYMBOL_SPECS.get(sym, SYMBOL_SPECS["XAUUSD"]))
    tf_kw = dict(PRESETS.get(tf, PRESETS["M5"]))
    base_tf = PRESETS["M5"]

    min_scale = float(tf_kw.get("min_sl_pips", base_tf["min_sl_pips"])) / float(base_tf["min_sl_pips"])
    max_scale = float(tf_kw.get("max_sl_pips", base_tf["max_sl_pips"])) / float(base_tf["max_sl_pips"])

    merged = dict(sym_kw)
    merged["htf_rule"] = tf_kw["htf_rule"]
    merged["min_sl_pips"] = round(float(sym_kw.get("min_sl_pips", 5.0)) * min_scale, 2)
    merged["max_sl_pips"] = round(float(sym_kw.get("max_sl_pips", 40.0)) * max_scale, 2)

    if overrides:
        merged.update(overrides)
    return StrategyParams(**merged)
