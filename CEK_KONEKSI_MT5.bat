@echo off
chcp 65001 >nul
title Diagnostic Koneksi MT5 Finex Demo
color 0E

echo ======================================================================
echo           DIAGNOSTIK KONEKSI METATRADER 5 FINEX DEMO
echo ======================================================================
echo.
cd /d "%~dp0finex-bot"

python export_history.py --check

echo.
echo ======================================================================
echo   Pemeriksaan selesai.
echo ======================================================================
pause
