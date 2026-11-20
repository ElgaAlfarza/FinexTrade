# FinexTrade (finex-bot): Sistem Trading Algoritmik & Risk Controller

Sistem trading algoritmik berbasis Python untuk MetaTrader 5 dan simulator Paper Trading lokal, dirancang dengan prinsip **fail-closed, transparansi kuantitatif, dan perlindungan modal**.

---

## ⚠️ Peringatan Risiko Resmi (Risk Disclaimer)

> **PERINGATAN RISIKO TINGGI:**
> Trading valuta asing (Forex) dan komoditas (khususnya Gold / XAUUSD) menggunakan leverage melibatkan risiko kerugian yang sangat tinggi dan tidak cocok untuk semua investor. Tingginya leverage dapat merugikan Anda sama besarnya dengan menguntungkan Anda.
> 
> Perangkat lunak ini adalah **alat bantu teknis dan otomasi pengujian kuantitatif**, BUKAN penasihat keuangan (financial advisor) dan **TIDAK memberikan jaminan profit dalam bentuk apa pun**. Kinerja masa lalu atau hasil backtest tidak menjamin hasil di masa depan. Seluruh risiko finansial berada sepenuhnya pada tanggung jawab pengguna.

---

## 1. Arsitektur & Strategi yang Aktif

- **Strategi Utama Live:** `EMA_TREND`
  - **Indikator:** Exponential Moving Average (EMA 8 / 21 / 50 Ribbon), Average Directional Index (ADX 14, ambang batas $\ge 20$ anti-sideways), Relative Strength Index (RSI 14, filter exhaustion $32 - 68$).
  - **Dynamic Retest Zone:** Pullback harga ke kantong dinamis antara EMA 8 dan EMA 21.
  - **Konfirmasi Candle:** Rejection wick $\ge 35\%$ dari candle range atau candle impulsif searah tren dengan range minimal $0.45 \times \text{ATR}$.
  - **Room-to-Move Filter:** Jarak minimal $1.0 \times \text{ATR}$ ke swing support/resistance 20 bar terakhir.
- **Strategi Sekunder (Backtest):** Fibonacci Swing Structure & Break-Even (`strategy.py`).
- **Platform Eksekusi:** MetaTrader 5 Desktop (IPC) & Paper Simulator lokal terisolasi.
- **Default Keamanan:** Hanya aktif di mode **DEMO** dan **PAPER**. Order ke akun REAL diblokir secara fail-closed kecuali disetel `ALLOW_LIVE_REAL=1` secara eksplisit. Akun Cent (CNT) ditolak hingga didukung penuh.

---

## 2. Fitur Keamanan & Perlindungan Modal

1. **Autentikasi & Proteksi API:**
   - Token sesi acak kriptografis (`SESSION_TOKEN`) dihasilkan setiap server dimulai dan wajib disertakan via header `X-Session-Token` atau parameter query WebSocket.
   - CORS dibatasi ketat ke `http://127.0.0.1:8000` dan `http://localhost:8000`. Request tanpa token ditolak `401 Unauthorized`.
   - Rate limiting ketat untuk endpoint eksekusi order guna mencegah serangan spam/rebinding.
2. **Circuit Breaker Harian Persisten (SQLite):**
   - Kerugian harian dihitung dalam unit $R$ dan disimpan secara atomic ke SQLite `data/bot_state.db`.
   - Akumulasi kerugian harian bertahan jika bot crash atau di-restart. Batas default: $-3.0R$.
3. **Monotonic Profit Lock Guard:**
   - Penguncian profit berjenjang non-decreasing yang memantau pergerakan harga running dan mengamankan cuan sebelum tergerus balik minus.
4. **News Shield (Fail-Closed):**
   - Kalender ekonomi otomatis menahan order sebelum dan sesudah rilis berita berdampak tinggi (High-Impact USD).
   - Bila data kalender kedaluwarsa ($> 24$ jam) atau offline, sistem otomatis menahan entri demi keselamatan akun.
5. **Ketiadaan AI di Jalur Eksekusi:**
   - Seluruh logika sinyal dan manajemen risiko 100% deterministik tanpa ketergantungan keputusan ke LLM (Gemini).

---

## 3. Validasi Kuantitatif Out-of-Sample (Data Riil XAUUSD)

Hasil evaluasi walk-forward jujur pada data historis broker Finex XAUUSD M5 (25.000 candle):

| Parameter | In-Sample (70% Data Awal) | Out-of-Sample (30% Data Uji) |
| :--- | :--- | :--- |
| **Rentang Waktu** | 11 Ags 2026 s/d 11 Okt 2026 | 11 Okt 2026 s/d 06 Nov 2026 |
| **Jumlah Trade** | 208 trade | 72 trade |
| **Win Rate** | **11.1%** (95% CI: 7.5% - 16.0%) | **4.2%** (95% CI: 1.4% - 11.5%) |
| **Profit Factor** | 0.08 | 0.02 |
| **Expectancy** | **-0.710 R** per trade | **-0.842 R** per trade |
| **Max Drawdown** | -77.3% | -45.6% |

### Analisis Kuantitatif Mengapa Hasil Scalping Menurun:
Target scalping ketat ($5 / 5 pips) pada instrumen Emas (XAUUSD) tergerus oleh friksi biaya transaksi riil:
- Spread rata-rata: 1.8 pip
- Slippage estimasi: 0.5 pip
- **Total Friksi Biaya: 2.3 pip (46% dari target profit 5 pip)**

Mengejar target 5 pip dengan biaya 2.3 pip secara matematis menghasilkan ekspektasi negatif setelah spread dan slippage. Sistem **menghapus semua klaim tidak berdasar** (seperti "Akurasi 69.1%" atau "Target Win Rate 90%") dan menampilkan fakta statistik sebenarnya.

---

## 4. Cara Menjalankan

### Persyaratan:
- Python 3.11+ (Disarankan Python 3.11 s/d 3.14)
- Windows OS (untuk MetaTrader 5 IPC)

### Instalasi Dependensi:
```bash
pip install -r requirements.txt
```

### Menjalankan Server & Dashboard:
```bash
python finex-bot/run_dashboard.py
```
Buka peramban di: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Menjalankan Evaluasi Kuantitatif Out-of-Sample:
```bash
python finex-bot/run_oos_evaluation.py
```

### Menjalankan Selftest Arsitektur:
```bash
python finex-bot/selftest.py
```

### Menjalankan Seluruh Pengujian Otomatis (Pytest):
```bash
python -m pytest tests/ -v
```

---

## 5. Struktur Berkas Utama

- `finex-bot/server.py`: API Server FastAPI, WebSocket telemetry, autentikasi sesi token, CORS guard, dan rate limiter.
- `finex-bot/live_engine.py`: Mesin utama live bot trading dengan strategi `EMA_TREND`, News Shield terintegrasi, dan fail-closed guard.
- `finex-bot/execution.py`: Engine eksekusi order (`MT5Executor` dan `PaperExecutor` thread-safe), Circuit Breaker persisten SQLite, dan Monotonic Profit Lock.
- `finex-bot/strategy.py`: Generator sinyal deterministik `EMA_TREND` dan Fibonacci.
- `finex-bot/backtest.py`: Engine backtest konservatif dengan pemodelan spread, slippage, Wilson 95% CI, dan walk-forward splitting.
- `finex-bot/migrate_trades.py`: Alat sanitasi CSV 12-kolom dan migrasi ke SQLite table `trades`.
- `finex-bot/data/bot_state.db`: Database SQLite untuk state circuit breaker dan seluruh jurnal trade terstruktur.
- `tests/`: Rangkaian unit test komprehensif fase 0 hingga 5 (33 pengujian otomatis).
