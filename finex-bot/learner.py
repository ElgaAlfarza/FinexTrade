"""Pembelajar online terkalibrasi (L2-Regularized Logistic Regression + Welford Z-Scaling).

Mencatat fitur setiap trade tertutup, menormalisasi skala fitur secara online (sehingga fitur
seperti `spread` tidak mendominasi fitur rasio seperti `wick` atau `zone_depth`), dan baru aktif
menyaring sinyal berprobabilitas rendah setelah minimal MIN_SAMPLES trade terkumpul.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict

import numpy as np

STATE = "learner_state.json"
FEATS = ["trend", "zone_depth", "wick", "rr", "sl_atr", "hour_sin", "hour_cos", "spread"]
MIN_SAMPLES = 30
THRESHOLD = 0.45
LR = 0.04
L2_DECAY = 0.015


class Learner:
    def __init__(self, path: str = STATE):
        self.path = path
        d_len = len(FEATS)
        self.w = np.zeros(d_len + 1, dtype=float)
        self.mean = np.zeros(d_len, dtype=float)
        self.m2 = np.ones(d_len, dtype=float)  # Welford akumulator varians (prior 1.0 agar aman di awal)
        self.n = 0
        self.wins = 0
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                if len(d.get("w", [])) == d_len + 1:
                    self.w = np.asarray(d["w"], dtype=float)
                if len(d.get("mean", [])) == d_len:
                    self.mean = np.asarray(d["mean"], dtype=float)
                if len(d.get("m2", [])) == d_len:
                    self.m2 = np.asarray(d["m2"], dtype=float)
                self.n = int(d.get("n", 0))
                self.wins = int(d.get("wins", 0))
            except Exception:
                pass

    @staticmethod
    def raw_vec(f: Dict[str, float]) -> np.ndarray:
        vals = []
        for k in FEATS:
            v = float(f.get(k, 0.0) or 0.0)
            vals.append(v if math.isfinite(v) else 0.0)
        return np.asarray(vals, dtype=float)

    def _std(self) -> np.ndarray:
        denom = max(self.n, 1)
        return np.sqrt(np.maximum(self.m2 / denom, 1e-4))

    def vec(self, f: Dict[str, float]) -> np.ndarray:
        raw = self.raw_vec(f)
        if self.n >= 5:
            z = np.clip((raw - self.mean) / self._std(), -4.0, 4.0)
        else:
            # Skala prior wajar sebelum 5 sampel pertama terkumpul
            prior_scale = np.array([1.0, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 3.0], dtype=float)
            z = np.clip(raw / prior_scale, -4.0, 4.0)
        return np.r_[1.0, z]

    def predict(self, f: Dict[str, float]) -> float:
        z = float(np.clip(self.w @ self.vec(f), -25.0, 25.0))
        return 1.0 / (1.0 + math.exp(-z))

    @property
    def ready(self) -> bool:
        return self.n >= MIN_SAMPLES

    def update(self, f: Dict[str, float], win: bool) -> None:
        raw = self.raw_vec(f)
        # Update statistik normalisasi Welford secara kausal
        self.n += 1
        delta = raw - self.mean
        self.mean += delta / self.n
        delta2 = raw - self.mean
        self.m2 += delta * delta2

        x = self.vec(f)
        target = 1.0 if win else 0.0
        for _ in range(3):
            p = 1.0 / (1.0 + math.exp(-float(np.clip(self.w @ x, -25.0, 25.0))))
            reg = np.r_[0.0, L2_DECAY * self.w[1:]]  # regularisasi L2 pada bobot fitur (bukan bias)
            self.w += LR * ((target - p) * x - reg)

        self.wins += int(bool(win))
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "w": self.w.tolist(),
                        "mean": self.mean.tolist(),
                        "m2": self.m2.tolist(),
                        "n": self.n,
                        "wins": self.wins,
                    },
                    fh,
                )
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            "samples": self.n,
            "win_rate": round(self.wins / self.n * 100.0, 1) if self.n else 0.0,
            "active": self.ready,
            "threshold": THRESHOLD,
        }
