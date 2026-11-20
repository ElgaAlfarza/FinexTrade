"""Unit test untuk Fase 0: Pengamanan Kredensial, Autentikasi API, dan Guard Akun Real."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Pastikan finex-bot ada di PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
FINEX_DIR = BASE_DIR / "finex-bot"
if str(FINEX_DIR) not in sys.path:
    sys.path.insert(0, str(FINEX_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tests.mock_mt5 import MockMT5, MockAccountInfo


def test_secret_scanner():
    """Memastikan tidak ada password plaintext atau kredensial yang ter-hardcode di file Python produksi."""
    forbidden_patterns = [
        re.compile(r'["\']password["\']\s*:\s*["\'][^"\']{4,}["\']', re.IGNORECASE),
        re.compile(r'["\']investor_password["\']\s*:\s*["\'][^"\']{4,}["\']', re.IGNORECASE),
    ]

    scanned_files = []
    found_violations = []

    for root, dirs, files in os.walk(FINEX_DIR):
        # Abaikan __pycache__, scratch, archive
        if any(ignored in root for ignored in ["__pycache__", "scratch", "archive", "data"]):
            continue
        for f in files:
            if f.endswith(".py"):
                fp = Path(root) / f
                scanned_files.append(fp)
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    for line_num, line in enumerate(fh, 1):
                        for pattern in forbidden_patterns:
                            if pattern.search(line):
                                found_violations.append(f"{fp.relative_to(BASE_DIR)}:{line_num} -> {line.strip()[:60]}")

    assert len(scanned_files) >= 10, f"Harus memindai minimal 10 file python produksi, hanya {len(scanned_files)}"
    assert not found_violations, f"Ditemukan password plaintext di file produksi:\n" + "\n".join(found_violations)


def test_mt5_session_real_account_guard(monkeypatch):
    """Memastikan MT5Session menolak akun REAL jika ALLOW_LIVE_REAL=0."""
    from mt5_client import MT5Session
    mock_mt5 = MockMT5()
    # Atur trade_mode ke REAL (2)
    mock_mt5._account.trade_mode = MockMT5.ACCOUNT_TRADE_MODE_REAL

    monkeypatch.setattr("mt5_client._import_mt5", lambda: mock_mt5)
    monkeypatch.setenv("ALLOW_LIVE_REAL", "0")

    session = MT5Session(login=123456, password="", server="Broker-Real", require_demo=True)
    with pytest.raises(RuntimeError, match="BUKAN akun demo|REAL terdeteksi"):
        session.__enter__()


def test_mt5_session_cent_account_guard(monkeypatch):
    """Memastikan MT5Session menolak akun Cent (CNT / USDC)."""
    from mt5_client import MT5Session
    mock_mt5 = MockMT5()
    mock_mt5._account.currency = "CNT"

    monkeypatch.setattr("mt5_client._import_mt5", lambda: mock_mt5)
    monkeypatch.setenv("ALLOW_LIVE_REAL", "0")

    session = MT5Session(login=123456, password="", server="Broker-Cent", require_demo=True)
    with pytest.raises(RuntimeError, match="berdenominasi Cent"):
        session.__enter__()


def test_mt5_executor_order_real_guard(monkeypatch):
    """Memastikan MT5Executor.open_trade memblokir pengiriman order ke akun REAL."""
    from execution import MT5Executor
    from mt5_client import MT5Session

    mock_mt5 = MockMT5()
    mock_mt5._account.trade_mode = MockMT5.ACCOUNT_TRADE_MODE_REAL

    session = MT5Session(login=123456, password="", server="Broker-Real", require_demo=False)
    session.mt5 = mock_mt5
    executor = MT5Executor(session)

    monkeypatch.setenv("ALLOW_LIVE_REAL", "0")
    pos = executor.open_trade("XAUUSD", "BUY", 0.01, sl=2640.0, tp=2660.0, risk_pips=20.0)

    assert pos is None
    assert "REAL diblokir" in executor.last_order_error
    assert len(mock_mt5.sent_orders) == 0


def test_api_unauthorized_without_token():
    """Memastikan request API tanpa token ditolak dengan HTTP 401."""
    from server import app
    client = TestClient(app)

    res = client.get("/api/status")
    assert res.status_code == 401
    assert "Unauthorized" in res.json().get("detail", "")


def test_api_authorized_with_token():
    """Memastikan request API dengan token sesi yang benar diterima."""
    from server import app, SESSION_TOKEN
    client = TestClient(app)

    res = client.get("/api/status", headers={"X-Bot-Token": SESSION_TOKEN})
    assert res.status_code == 200


def test_api_cross_origin_blocked():
    """Memastikan request dengan Origin eksternal / tak dikenal ditolak (CORS/Origin Protection)."""
    from server import app, SESSION_TOKEN
    client = TestClient(app)

    res = client.get("/api/status", headers={
        "X-Bot-Token": SESSION_TOKEN,
        "Origin": "http://evil-tracker-site.com"
    })
    assert res.status_code == 403


def test_frontend_injects_session_token():
    """Memastikan endpoint root GET / menyajikan token sesi ke konteks frontend."""
    from server import app, SESSION_TOKEN
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    assert "window.__BOT_AUTH_TOKEN__" in res.text
    assert SESSION_TOKEN in res.text
