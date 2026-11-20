import os
import sys
import threading
import time
import webbrowser
import uvicorn

def open_browser():
    time.sleep(1.5)
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass

if __name__ == "__main__":
    print("=====================================================")
    print("  FINEX BOT - FULL-STACK TRADING TERMINAL V2.0")
    print("  Dashboard Browser: http://localhost:8000")
    print("  Status: Server Aktif & Realtime Telemetri Siap")
    print("=====================================================")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
