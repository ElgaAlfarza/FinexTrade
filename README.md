# FinexTrade (finex-bot): Sistem Trading Algoritmik & Risk Controller

Silakan baca dokumentasi lengkap proyek di [finex-bot/README.md](file:///d:/BigBoz/FinexTrade/finex-bot/README.md).

Sistem trading algoritmik berbasis Python untuk MetaTrader 5 dan Paper Trading simulator dengan arsitektur fail-closed, transparansi kuantitatif, dan perlindungan modal.

### Menjalankan Pengujian:
```bash
python -m pytest tests/ -v
python finex-bot/selftest.py
python finex-bot/run_oos_evaluation.py
```
