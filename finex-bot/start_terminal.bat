@echo off
title Finex Bot Pro Terminal
echo ========================================================
echo   FINEX BOT TRADING TERMINAL - FULL-STACK EDITION
echo ========================================================
echo.
cd /d "%~dp0"
echo Memulai server dashboard di http://localhost:8000 ...
python run_dashboard.py
pause
