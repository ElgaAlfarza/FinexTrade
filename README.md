# FinexTrade: Quantitative Algorithmic Trading & Risk Control Terminal

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/pytest-38%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/architecture-Fail--Closed%20%26%20Zero--Trust-orange.svg)]()

Sistem trading algoritmik berbasis Python berstandar institusional untuk MetaTrader 5 (MT5 Direct IPC) dan Paper Trading Simulator lokal terisolasi. Dibangun dengan fokus utama pada **prinsip deterministik, fail-closed security, integritas data kuantitatif, dan perlindungan modal (risk preservation)**.

---

## ⚠️ Peringatan Risiko Resmi (Risk Disclaimer)

> **PERINGATAN RISIKO TINGGI:**
> Trading valuta asing (Forex) dan komoditas (khususnya Gold / XAUUSD) menggunakan leverage melibatkan risiko kerugian yang sangat tinggi dan tidak cocok untuk semua investor. Tingginya leverage dapat merugikan Anda sama besarnya dengan menguntungkan Anda.
> 
> Proyek ini dirancang sebagai **portofolio rekayasa perangkat lunak finansial (fintech), riset kuantitatif, dan otomasi pengujian strategi**, BUKAN penasihat keuangan (*financial advisor*) dan **TIDAK memberikan jaminan profit dalam bentuk apa pun**. Kinerja masa lalu atau hasil backtest tidak menjamin hasil di masa depan.

---

## 🌟 Fitur Utama & Keunggulan Arsitektur

### 1. Eksekusi Deterministik & Tanpa Halusinasi AI
- Seluruh logika sinyal (`EMA_TREND`, Fibonacci Retracement, Dynamic Retest Zone, ADX Filter) dieksekusi secara 100% deterministik dengan perhitungan matematika `pandas` dan `numpy`.
- Model LLM (seperti Google Gemini) **hanya digunakan untuk modul riset/analisis sentimen opsional** di luar jalur kritis dan dilarang keras memicu eksekusi order secara langsung.

### 2. Perlindungan Modal Bertingkat (Capital Guardrails)
- **Persistent Daily Circuit Breaker (SQLite):** Kerugian harian dihitung dalam unit $R$ dan disimpan secara atomik ke basis data SQLite. Batas default $-3.0R$ tetap aktif dan bertahan meski server/bot mengalami *restart* atau *crash*.
- **Monotonic Profit Lock Guard:** Sistem penguncian profit berjenjang non-decreasing yang mengunci keuntungan saat pergerakan harga menguntungkan agar tidak berbalik menjadi kerugian.
- **News Shield (Fail-Closed):** Penahan order otomatis sebelum dan sesudah rilis data ekonomi berdampak tinggi (High-Impact USD). Bila kalender *offline* atau kedaluwarsa ($> 24$ jam), sistem otomatis menahan eksekusi demi keselamatan modal.
- **Fail-Closed Real Account Guard:** Mode eksekusi secara *default* terkunci pada mode **DEMO** atau **PAPER SIMULATOR**. Order ke akun REAL diblokir keras di level kode (`ALLOW_LIVE_REAL=0`).

### 3. Keamanan Kredensial & Autentikasi API
- **Token Sesi Kriptografis (`SESSION_TOKEN`):** Dihasilkan secara acak saat startup dan wajib disertakan via header `X-Session-Token` atau query parameter WebSocket.
- **CORS & Rate Limiter:** Pembatasan origin ketat dan pembatasan frekuensi order untuk memitigasi risiko serangan *request rebinding* atau *spamming*.
- **Brankas Kredensial Terisolasi:** Kredensial akun MT5 hanya dibaca dari *environment variables* lokal (`.env`), disanitasi dari log server, dan tidak pernah diekspos ke antarmuka peramban.

---

## 📊 Validasi Kuantitatif Out-of-Sample (Data Riil XAUUSD)

Hasil evaluasi walk-forward jujur pada data historis broker Finex XAUUSD M5 (25.000 candle):

| Parameter | In-Sample (70% Data Awal) | Out-of-Sample (30% Data Uji) |
| :--- | :--- | :--- |
| **Rentang Waktu** | 11 Ags 2026 s/d 11 Okt 2026 | 11 Okt 2026 s/d 06 Nov 2026 |
| **Jumlah Trade** | 208 trade | 72 trade |
| **Win Rate** | **11.1%** (95% CI: 7.5% - 16.0%) | **4.2%** (95% CI: 1.4% - 11.5%) |
| **Profit Factor** | 0.08 | 0.02 |
| **Expectancy** | **-0.710 R** per trade | **-0.842 R** per trade |
| **Max Drawdown** | -77.3% | -45.6% |

### Analisis Kuantitatif Friksi Biaya:
Target scalping ketat ($5 / 5$ pips) pada instrumen Emas (XAUUSD) tergerus secara signifikan oleh friksi biaya transaksi riil:
- Spread rata-rata: 1.8 pip
- Slippage estimasi: 0.5 pip
- **Total Friksi Biaya: 2.3 pip (46% dari target profit 5 pip)**

> **Pelajaran Kuantitatif:** Mengejar target 5 pip dengan friksi 2.3 pip secara matematis menghasilkan ekspektasi negatif setelah *spread* dan *slippage*. Sistem ini secara jujur memaparkan fakta statistik tanpa *overfitting* atau klaim palsu.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn, SQLite3, WebSocket (Real-time Telemetry)
- **Data & Quantitative Analysis:** Pandas, NumPy, Scikit-learn
- **Integrasi Broker:** MetaTrader 5 Python API (Direct IPC Windows)
- **Frontend Dashboard:** Modern Responsive Web UI (Dual Theme: Dark Neon Glass & Santorini Light), Charting Engine
- **Quality Assurance:** Pytest (38 Unit Tests mencakup keamanan, determinisme, risiko, jurnal, dan paritas backtest)

---

## 🚀 Panduan Memulai Cepat

### 1. Kloning Repositori & Instalasi
```bash
git clone https://github.com/<username>/FinexTrade.git
cd FinexTrade
pip install -r requirements.txt
```

### 2. Konfigurasi Environment
Salin template konfigurasi:
```bash
cp .env.example .env
```
Buka file `.env` dan masukkan konfigurasi akun MT5 Demo Anda. Jangan pernah meng-commit file `.env` ke Git!

### 3. Menjalankan Dashboard & Bot Engine
```bash
python run_dashboard.py
```
Akses dashboard interaktif melalui peramban di: **[http://localhost:8000](http://localhost:8000)**

---

## 🧪 Menjalankan Rangkaian Pengujian Otomatis

Seluruh logika arsitektur dan perlindungan risiko diverifikasi dengan pengujian otomatis:

```bash
# 1. Jalankan seluruh unit test (38 test)
python -m pytest tests/ -v

# 2. Jalankan selftest arsitektur dan simulasi end-to-end
python finex-bot/selftest.py

# 3. Jalankan evaluasi out-of-sample walk-forward
python finex-bot/run_oos_evaluation.py
```

---

## 📁 Struktur Direktori Proyek

```text
FinexTrade/
├── .env.example                     # Template konfigurasi aman (bebas rahasia)
├── .gitignore                        # Konfigurasi proteksi privasi & file sensitif
├── README.md                         # Dokumentasi portofolio utama
├── requirements.txt                  # Daftar dependensi teruji
├── run_dashboard.py                  # Entrypoint server & dashboard lokal
├── KLIK_DUA_KALI_UNTUK_JALANKAN.bat  # Launcher satu klik untuk Windows
├── CEK_KONEKSI_MT5.bat               # Diagnostik koneksi terminal MT5
├── finex-bot/                        # Modul inti mesin trading
│   ├── account_vault.py              # Modul brankas multi-akun aman
│   ├── backtest.py                   # Engine backtest & Wilson CI
│   ├── execution.py                  # Eksekutor order MT5 & Paper Simulator
│   ├── live_engine.py                # Loop pemantau pasar & strategi live
│   ├── news_radar.py                 # News Shield kalender ekonomi
│   ├── risk.py                       # Dynamic lot sizing & risk guards
│   ├── server.py                     # API FastAPI & WebSocket
│   ├── strategy.py                   # Logika sinyal deterministik
│   └── web/
│       └── index.html                # Antarmuka web dashboard interaktif
└── tests/                            # Rangkaian unit test (Fase 0 - Fase 5)
    ├── mock_mt5.py                   # Mock driver MT5 terisolasi
    ├── test_phase0_security.py       # Pengujian proteksi kredensial & API token
    ├── test_phase1_determinism.py    # Pengujian anti-repaint & time determinism
    ├── test_phase2_honesty.py        # Pengujian transparansi & penghapusan gate palsu
    ├── test_phase3_risk.py           # Pengujian circuit breaker & profit lock
    ├── test_phase4_journal.py        # Pengujian integritas jurnal & SQLite
    ├── test_phase5_backtest.py       # Pengujian paritas backtest
    └── test_quick_order.py           # Pengujian validasi endpoint order
```

---

## 📜 Lisensi & Atribusi

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE).
Dikembangkan untuk tujuan riset kuantitatif dan demonstrasi portofolio rekayasa perangkat lunak finansial.
