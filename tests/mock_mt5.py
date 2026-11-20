"""Mock modul MetaTrader5 untuk unit testing independen."""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class MockAccountInfo:
    def __init__(self, login=61950031, server="FinexBisnisSolusi-Demo", currency="USD",
                 balance=10000.0, equity=10000.0, margin_free=10000.0, trade_mode=0,
                 leverage=500, trade_allowed=True, trade_expert=True):
        self.login = login
        self.server = server
        self.currency = currency
        self.balance = balance
        self.equity = equity
        self.margin_free = margin_free
        self.trade_mode = trade_mode  # 0=DEMO, 1=CONTEST, 2=REAL
        self.leverage = leverage
        self.trade_allowed = trade_allowed
        self.trade_expert = trade_expert


class MockTerminalInfo:
    def __init__(self, trade_allowed=True, connected=True):
        self.trade_allowed = trade_allowed
        self.connected = connected


class MockSymbolInfo:
    def __init__(self, symbol="XAUUSD", point=0.01, digits=2, stops_level=0,
                 volume_min=0.01, volume_step=0.01, volume_max=100.0):
        self.name = symbol
        self.point = point
        self.digits = digits
        self.trade_stops_level = stops_level
        self.volume_min = volume_min
        self.volume_step = volume_step
        self.volume_max = volume_max
        self.margin_initial = 0.0
        self.trade_contract_size = 100.0
        self.currency_base = "XAU"
        self.currency_profit = "USD"


class MockTick:
    def __init__(self, bid=2650.00, ask=2650.30, time_val=1700000000):
        self.bid = bid
        self.ask = ask
        self.time = time_val


class MockOrderResult:
    def __init__(self, retcode=10009, comment="Request executed"):
        self.retcode = retcode
        self.order = 999001
        self.deal = 888001
        self.volume = 0.01
        self.price = 2650.30
        self.comment = comment


class MockMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    TRADE_RETCODE_DONE = 10009

    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2

    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 16385
    TIMEFRAME_D1 = 16408

    def __init__(self):
        self._initialized = False
        self._account = MockAccountInfo()
        self._terminal = MockTerminalInfo()
        self._ticks: Dict[str, MockTick] = {
            "XAUUSD": MockTick(2650.00, 2650.30),
            "EURUSD": MockTick(1.08500, 1.08510),
        }
        self._symbols: Dict[str, MockSymbolInfo] = {
            "XAUUSD": MockSymbolInfo("XAUUSD", 0.01, 2, 0, 0.01, 0.01, 100.0),
            "EURUSD": MockSymbolInfo("EURUSD", 0.00001, 5, 0, 0.01, 0.01, 100.0),
        }
        self.sent_orders: List[Any] = []
        self._last_error = (1, "Success")

    def initialize(self, **kwargs) -> bool:
        self._initialized = True
        return True

    def shutdown(self) -> None:
        self._initialized = False

    def login(self, login, password="", server="") -> bool:
        self._account.login = login
        self._account.server = server
        return True

    def account_info(self) -> Optional[MockAccountInfo]:
        return self._account

    def terminal_info(self) -> Optional[MockTerminalInfo]:
        return self._terminal

    def symbol_info(self, symbol: str) -> Optional[MockSymbolInfo]:
        return self._symbols.get(symbol, MockSymbolInfo(symbol))

    def symbol_info_tick(self, symbol: str) -> Optional[MockTick]:
        return self._ticks.get(symbol, MockTick())

    def symbol_select(self, symbol: str, select: bool = True) -> bool:
        return True

    def last_error(self) -> Tuple[int, str]:
        return self._last_error

    def order_send(self, request: Any) -> MockOrderResult:
        self.sent_orders.append(request)
        return MockOrderResult()

    def positions_get(self, symbol: Optional[str] = None, ticket: Optional[int] = None) -> Tuple[Any, ...]:
        return ()

    def history_deals_get(self, *args, **kwargs) -> Tuple[Any, ...]:
        return ()

    def copy_rates_from_pos(self, symbol: str, tf: int, start_pos: int, count: int) -> np.ndarray:
        dtype = [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i4"),
            ("real_volume", "i8"),
        ]
        arr = np.zeros(count, dtype=dtype)
        base = 2650.0 if "XAU" in symbol else 1.0850
        for i in range(count):
            arr[i] = (1700000000 + i * 60, base, base + 1.0, base - 1.0, base + 0.2, 100, 20, 100)
        return arr
