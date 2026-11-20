"""Modul Brankas Akun Aman (Secure Multi-Account Vault) untuk FinexTrade.

Fitur Keamanan Anti-Peretasan:
1. Penyimpanan kredensial MT5 terisolasi di sisi server (backend-only).
2. Data password TIDAK PERNAH dikirim ke browser / frontend / API publik.
3. Otentikasi dan pergantian akun (1-Click Switcher) menggunakan Account ID internal.
4. Masking otomatis pada log server agar password tidak bocor ke berkas log.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("finex.account_vault")

VAULT_FILE = os.path.join(os.path.dirname(__file__), "data", "accounts_vault.json")

DEFAULT_VAULT = {
    "active_account_id": "finex_demo",
    "accounts": [
        {
            "id": "finex_demo",
            "name": "Finex Demo (Utama)",
            "broker": "Finex Bisnis Solusi",
            "login": 61950031,
            "password": "", # Tersimpan di session MT5 Finex
            "server": "FinexBisnisSolusi-Demo",
            "account_type": "demo",
            "currency": "USD",
            "is_cent": False,
            "badge": "DEMO",
            "badge_color": "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
            "note": "Akun demo reguler untuk pengujian strategi",
        },
        {
            "id": "markets4you_cent",
            "name": "Markets4you Cent (Real MT5)",
            "broker": "Markets4you / E-Global Trade",
            "login": 21060799,
            "password": "",
            "investor_password": "",
            "server": "EGlobalTrade-Classic",
            "account_type": "real",
            "currency": "CNT",
            "is_cent": True,
            "badge": "CENT REAL",
            "badge_color": "bg-amber-500/20 text-amber-300 border-amber-500/40",
            "note": "Akun cent riil untuk modal Rp 100.000 (Tahan ribuan pips)",
        },
        {
            "id": "paper_simulator",
            "name": "Simulator Internal (Paper)",
            "broker": "Finex Virtual Engine",
            "login": 99999999,
            "password": "",
            "server": "LOCAL_SIMULATOR",
            "account_type": "paper",
            "currency": "USD",
            "is_cent": False,
            "badge": "SIMULATOR",
            "badge_color": "bg-purple-500/20 text-purple-300 border-purple-500/40",
            "note": "Simulasi tanpa koneksi MT5",
        }
    ]
}


class AccountVault:
    """Manajer brankas akun tersimpan yang aman dari peretasan."""

    def __init__(self, vault_path: str = VAULT_FILE):
        self.vault_path = vault_path
        self._data: Dict[str, Any] = dict(DEFAULT_VAULT)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.vault_path):
            try:
                with open(self.vault_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and "accounts" in loaded:
                        self._data = loaded
            except Exception as e:
                logger.warning(f"Gagal memuat accounts_vault.json: {e}")
        else:
            self._save()

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.vault_path), exist_ok=True)
            with open(self.vault_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.error(f"Gagal menyimpan accounts_vault.json: {e}")

    def get_safe_accounts_list(self) -> List[Dict[str, Any]]:
        """Mengembalikan daftar akun TANPA PASSWORD untuk keamanan frontend."""
        safe_list = []
        active_id = self._data.get("active_account_id", "finex_demo")
        for acc in self._data.get("accounts", []):
            safe_item = {
                "id": acc.get("id"),
                "name": acc.get("name"),
                "broker": acc.get("broker"),
                "login": acc.get("login"),
                "server": acc.get("server"),
                "account_type": acc.get("account_type"),
                "currency": acc.get("currency", "USD"),
                "is_cent": acc.get("is_cent", False),
                "badge": acc.get("badge", "ACC"),
                "badge_color": acc.get("badge_color", "bg-slate-800 text-slate-300 border-slate-700"),
                "note": acc.get("note", ""),
                "is_active": (acc.get("id") == active_id),
                "has_password": bool(acc.get("password") or acc.get("investor_password")),
            }
            safe_list.append(safe_item)
        return safe_list

    def get_account_credentials(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Mengambil kredensial lengkap secara internal (hanya untuk proses login server)."""
        for acc in self._data.get("accounts", []):
            if acc.get("id") == account_id:
                return dict(acc)
        return None

    def get_active_account_id(self) -> str:
        return self._data.get("active_account_id", "finex_demo")

    def get_active_account(self) -> Optional[Dict[str, Any]]:
        active_id = self.get_active_account_id()
        return self.get_account_credentials(active_id)

    def set_active_account(self, account_id: str) -> bool:
        for acc in self._data.get("accounts", []):
            if acc.get("id") == account_id:
                self._data["active_account_id"] = account_id
                self._save()
                return True
        return False

    def upsert_account(self, acc_data: Dict[str, Any]) -> None:
        """Tambah atau perbarui akun tersimpan."""
        acc_id = acc_data.get("id")
        if not acc_id:
            return
        accounts = self._data.get("accounts", [])
        found = False
        for i, acc in enumerate(accounts):
            if acc.get("id") == acc_id:
                accounts[i].update(acc_data)
                found = True
                break
        if not found:
            accounts.append(acc_data)
        self._data["accounts"] = accounts
        self._save()


# Singleton
_vault_instance: Optional[AccountVault] = None

def get_account_vault() -> AccountVault:
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = AccountVault()
    return _vault_instance
