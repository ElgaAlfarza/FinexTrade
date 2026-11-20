@echo off
chcp 65001 >nul
title Finex Algorithmic Trading Terminal v2.0
color 0B

echo ======================================================================
echo          FINEX TRADING BOT - INSTITUTIONAL TERMINAL PRO
echo ======================================================================
echo.
echo [1/3] Memeriksa direktori kerja...
cd /d "%~dp0"
echo     Direktori: %CD%
echo.

echo [2/3] Memeriksa Python environment...
set PY_CMD=
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=python
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PY_CMD=py
    )
)

if "%PY_CMD%"=="" (
    color 0C
    echo [ERROR] Python tidak terdeteksi di PATH sistem Windows!
    echo Silakan install Python 3.10+ atau tambahkan Python ke PATH.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('%PY_CMD% --version 2^>^&1') do set PY_VER=%%v
echo     Terdeteksi: %PY_VER%
echo.

echo [3/3] Memulai Server Dashboard & Bot Engine...
echo     URL Dashboard : http://localhost:8000
echo     (Browser Anda akan terbuka secara otomatis dalam 2 detik)
echo.
echo ======================================================================
echo   LOG AKTIFITAS BOT & SERVER (Tekan Ctrl+C untuk menghentikan):
echo ======================================================================
echo.

%PY_CMD% run_dashboard.py

echo.
echo ======================================================================
echo   Server dihentikan.
echo ======================================================================
pause
