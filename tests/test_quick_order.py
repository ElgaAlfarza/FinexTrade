import sys
import os
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "finex-bot"))
from server import app, SESSION_TOKEN, engine


@pytest.fixture(autouse=True)
def ensure_paper_mode():
    engine.use_paper = True
    time.sleep(0.1)


def test_quick_order_success():
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-Bot-Token": SESSION_TOKEN}
    
    # Wait for rate limit window if any
    time.sleep(1.05)
    
    payload = {
        "symbol": "XAUUSD",
        "side": "BUY",
        "lots": 0.01,
        "sl": 2500.0,
        "tp": 2800.0,
    }
    res = client.post("/api/trades/quick-order", json=payload, headers=headers)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["success"] is True
    assert "position" in data
    assert data["position"]["symbol"] == "XAUUSD"
    assert data["position"]["side"] == "BUY"
    assert data["position"]["lots"] == 0.01
    assert data["position"]["sl"] == 2500.0
    assert data["position"]["tp"] == 2800.0


def test_quick_order_rate_limiting():
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-Bot-Token": SESSION_TOKEN}
    
    time.sleep(1.05)
    # First order
    res1 = client.post("/api/trades/quick-order", json={"symbol": "XAUUSD", "side": "BUY", "lots": 0.01}, headers=headers)
    assert res1.status_code == 200
    
    # Immediate second order should be rate limited (429)
    res2 = client.post("/api/trades/quick-order", json={"symbol": "XAUUSD", "side": "SELL", "lots": 0.01}, headers=headers)
    assert res2.status_code == 429
    data = res2.json()
    assert "detail" in data
    assert "Harap tunggu 1 detik" in data["detail"]


def test_quick_order_invalid_side():
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-Bot-Token": SESSION_TOKEN}
    
    time.sleep(1.05)
    res = client.post("/api/trades/quick-order", json={"symbol": "XAUUSD", "side": "HOLD", "lots": 0.01}, headers=headers)
    assert res.status_code == 400
    data = res.json()
    assert "detail" in data
    assert "HOLD" in data["detail"]


def test_quick_order_unauthorized():
    client = TestClient(app, raise_server_exceptions=False)
    # No auth header
    res = client.post("/api/trades/quick-order", json={"symbol": "XAUUSD", "side": "BUY", "lots": 0.01})
    assert res.status_code == 401
    data = res.json()
    assert "detail" in data


def test_global_exception_handler_returns_json():
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-Bot-Token": SESSION_TOKEN}

    # Add a temporary test endpoint that raises an unhandled error
    @app.get("/api/test-error")
    def trigger_error():
        raise RuntimeError("Simulated crash for exception handler test")

    res = client.get("/api/test-error", headers=headers)
    assert res.status_code == 500
    data = res.json()  # MUST be valid JSON, not HTML or plain text!
    assert data["success"] is False
    assert "Terjadi kesalahan server internal" in data["detail"]
    assert "Simulated crash" in data["detail"]
