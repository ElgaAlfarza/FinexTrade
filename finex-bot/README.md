# Finex Bot: Algorithmic Trading Terminal (Tahap 1 s/d 4)

Bot trading otomatis berbasis **Multi-Timeframe Market Structure**, **Fibonacci Retracement (Golden Pocket / OTE 50% - 78.6%)**, **Rejection Candle Action**, dilengkapi **Full-Stack Web Dashboard (FastAPI + WebSockets)**, **Engine Eksekusi Dual-Mode (MT5 Demo & Paper Trading Simulator)**, serta **Risk Management & Circuit Breaker**.

---

## Ringkasan Status Arsitektur

| Komponen | Status | Keterangan |
| :--- | :--- | :--- |
| **Tahap 1: Koneksi MT5 & Data** | **Selesai** | `mt5_client.py`, `export_history.py` (ekspor rates BID ke CSV) |
| **Tahap 2: Sinyal & Backtest** | **Selesai** | `strategy.py`, `backtest.py` (tanpa lookahead, interval 95% Wilson) |
| **Tahap 3: Engine Eksekusi** | **Selesai** | `execution.py`, `live_engine.py` (MT5 + Paper Simulator, Auto Break-Even) |
| **Tahap 4: Full-Stack Dashboard** | **Selesai** | `server.py`, `web/index.html` (Web terminal real-time, live chart, backtest studio) |
| **Data Pasar Realistis** | **Tersedia** | `data/EURUSD_M5.csv`, `EURUSD_M1.csv`, `EURUSD_H1.csv`, `XAUUSD_M5.csv` |

---

## Cara Menjalankan

### 1. Buka Web Dashboard (Rekomendasi)
Cukup jalankan:
```bash
python run_dashboard.py
```
atau klik dua kali file `start_terminal.bat`.
Buka browser di: **[http://localhost:8000](http://localhost:8000)**

**Fitur Dashboard:**
- **Live Terminal & Positions**: Pantau Balance, Equity, Floating PnL, margin, batas rugi harian (circuit breaker), serta daftar posisi aktif dengan tombol **[BE]** (Break-Even) dan **[Close]** instan.
- **Interactive Chart**: Grafik candlestick visual dengan overlay zona Fibonacci 50%-78.6%, Swing High/Low leg HTF, dan target TP (-0.27).
- **Backtest Studio**: Jalankan backtest langsung dari UI pada dataset M1, M5, H1 atau Gold (XAUUSD), atur slider risiko %, SL mode, Auto BE, dan lihat kurva pertumbuhan ekuitas serta drawdown secara interaktif.
- **Bot Control**: Tombol **START BOT**, **PAUSE**, dan **EMERGENCY KILL SWITCH** (likuidasi semua posisi seketika).

---

### 2. Jalankan Backtest Lewat Terminal (CLI)
```bash
# Backtest EURUSD M5 dengan mode Structure SL
python run_backtest.py --csv data/EURUSD_M5.csv --tf M5

# Backtest dengan Auto Break-Even (+1R) dan SL Candle
python run_backtest.py --csv data/EURUSD_M5.csv --tf M5 --sl-mode candle --be

# Backtest XAUUSD (Emas M5)
python run_backtest.py --csv data/XAUUSD_M5.csv --tf M5 --pip-size 0.1 --spread-pips 2.5
```

---

### 3. Koneksi ke MetaTrader 5 (Akun Demo Finex)
1. Buka terminal MT5 Finex di Windows, pastikan tombol **Algo Trading** di toolbar atas aktif (berwarna hijau).
2. Salin `.env.example` menjadi `.env` lalu isi kredensial akun demo:
   ```ini
   MT5_LOGIN=nomor_akun_demo
   MT5_PASSWORD=kata_sandi_demo
   MT5_SERVER=FinexBisnisSolusi-Demo
   ```
   *(Server alamat manual jika belum muncul di MT5: `prod-mt5-demo1.fnx.xmt.mx:443`)*
3. Uji koneksi:
   ```bash
   python export_history.py --check
   ```
4. Ekspor data candle historis dari broker:
   ```bash
   python export_history.py --symbol EURUSD --tf M5 --from 2025-01-01 --out data/EURUSD_M5.csv
   ```

*Catatan Trader: Jika terminal MT5 sedang ditutup atau belum terhubung, sistem otomatis beralih ke **Paper Trading Simulator**, sehingga seluruh fitur bot dan dashboard tetap dapat diuji secara penuh.*

---

### 4. Menjalankan Selftest Otomatis
Untuk membuktikan bahwa seluruh mekanika risiko, formula lot, ketiadaan lookahead bias, dan simulator paper trading valid:
```bash
python selftest.py
```

---

## Logika Strategi & Risk Management (Trader Blueprint)

1. **Multi-Timeframe Trend (HTF)**:
   - Timeframe entry di-resample ke timeframe tren (M1 &rarr; M15, M5 &rarr; M30, H1 &rarr; H4).
   - Menggunakan fractal swing pivots ($k=3$ bar).
   - Tren terdefinisi: $HH + HL =$ **Uptrend (Bull)**; $LH + LL =$ **Downtrend (Bear)**.
2. **Zona Retracement Fibonacci (Optimal Trade Entry)**:
   - Ditarik dari leg impuls terakhir.
   - Zona masuk: level **50.0% s/d 78.6%**.
3. **Pemicu Entry (Rejection Price Action)**:
   - Candle konfirmasi di timeframe entry harus menyentuh zona dan membentuk sumbu penolakan (wick ratio $\ge 50\%$) searah tren.
4. **Filter Moving Average (opsional, bawaan aktif untuk XAUUSD)**:
   - MA dihitung di timeframe tren (HTF) dari candle yang SUDAH tertutup (tanpa lookahead).
   - Parameter di `config.py`: `use_ma_filter`, `ma_type` (SMA/EMA), `ma_fast`, `ma_slow`, `ma_mode`.
   - Mode: `price_slow` (buy jika close HTF > MA lambat), `cross` (MA cepat > MA lambat), `price_both` (close di atas kedua MA).
   - Filter hanya MENGURANGI sinyal; tidak pernah menambah. Angka bawaan (EMA 21/50) adalah hipotesis awal, BUKAN setting dari video TikTok (setting video tidak tersedia). Uji di data XAUUSD asli.
   - CLI: `python run_backtest.py --csv data/XAUUSD_M5.csv --tf M5 --pip-size 0.1 --spread-pips 2.5 --be --ma --ma-mode price_both`
   - Dashboard: Backtest Studio punya kotak centang dan pilihan MA; chart menggambar garis MA dan status MA (BUY_ONLY / SELL_ONLY / NETRAL).
5. **Proteksi & Manajemen Risiko**:
   - **Risiko per trade**: Default $1\%$ dari ekuitas (dihitung dinamis via `risk.py`).
   - **Circuit Breaker Harian**: Jika total kerugian harian mencapai $-3R$, bot berhenti trading hari itu.
   - **Auto Break-Even**: Saat posisi mengambang $+1R$, Stop Loss digeser otomatis ke level entry $+ 0.5$ pip buffer untuk mengunci risiko.
   - **Target TP**: Extension $-0.27$ dari leg impuls, dengan syarat minimum Reward-to-Risk $1 : 1.5$.


---

## Mode Latihan (khusus akun DEMO / PAPER)

Untuk mengumpulkan data belajar lebih cepat, bot bisa entry jauh lebih sering walau tidak ada sinyal Fibonacci.
Aktifkan lewat panel **Mode Latihan** di dashboard (atau `POST /api/bot/explore`).

- **Laju entry:** setiap N candle tertutup (`explore_every_bars`). Di timeframe M1, N=1 berarti sekitar 1 entry per menit, N=5 berarti per 5 menit. Pilih timeframe M1 di dashboard.
- **Arah:** mengikuti tren HTF (dan MA bila aktif). Jika ranging, memakai arah candle terakhir. Arah tren dan MA yang bertentangan membuat entry dilewati.
- **SL/TP:** SL = 1.5 x ATR(14), dijepit ke batas SL simbol; TP = RR 1.5 x jarak SL.
- **Tidak ada filter sesi dan tidak ada filter model belajar**, supaya data mencakup setup baik dan buruk.
- **Model terpisah:** entry latihan dipelajari oleh `learner_state_explore.json`, tidak mencemari model strategi utama.
- **Pengaman yang tetap aktif:** spread guard, batas jarak SL, lot berdasarkan risiko 1%, maksimal posisi terbuka (default 3), dan circuit breaker harian (default -10R).
- **Guard akun:** ditolak otomatis jika MT5 terdeteksi akun REAL.
- Entry latihan bukan strategi. Hasilnya tidak boleh dibaca sebagai profit strategi.

## Perbaikan Circuit Breaker

Sebelumnya penghitung R harian (`daily_r_pnl`) tidak pernah diperbarui sehingga batas -3R tidak pernah aktif di live. Sekarang setiap trade yang tutup dicatat ke R harian.

---

## Catatan Penting (Jujur)

- Semua file di `data/` dibuat oleh `generate_sample_data.py` (random walk sintetis), BUKAN data pasar asli. Hasil backtest di sana hanya membuktikan kode berjalan, bukan bahwa strategi menguntungkan. Ekspor data asli dengan `export_history.py` sebelum menilai strategi atau filter MA.
- Zona Fibonacci kini 50% sampai 78.6% sesuai dokumentasi (sebelumnya `fib_bottom` di kode bernilai 0.886).
