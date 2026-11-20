"""Pelatihan Model Prediktif EURUSD Berbasis Machine Learning.

Diadaptasi dari notebook `forex-eurusd-predictive-model.ipynb`:
1. Rekayasa Fitur (Feature Engineering):
   - Multi-horizon Moving Average Ratios (EMA 8, 21, 50, 200).
   - Multi-horizon Volatility Ratios (Rolling Standard Deviation 8, 21, 50).
   - Return Momentum (Returns 1, 5, 21 bars) & Volume Dynamics.
   - Microstructure wick & body pressure.
2. Formulasi Kuantitatif:
   - Target Regresi: Prediksi Drift Harga 5 Candle Mendatang (t+5).
   - Target Klasifikasi: Peluang Menang Arah (Directional Edge) dengan Rasio RR 1 : 1.6+.
3. Evaluasi Ketat:
   - 90% Data Latih (Chronological) vs 10% Data Uji (Out-of-sample, Tanpa Lookahead).
"""
import os
import json
import math
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "EURUSD_M5.csv")
OUTPUT_STATE = os.path.join(os.path.dirname(__file__), "data", "eurusd_model_state.json")


def generate_eurusd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 32 features for EURUSD adapted from the notebook's feature methodology."""
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["tick_volume"].astype(float) if "tick_volume" in df.columns else df["volume"].astype(float)
    pip = 0.0001  # 1 pip EURUSD

    feats = pd.DataFrame(index=df.index)

    # 1. Price Lags
    feats["open_1"] = o.shift(1)
    feats["close_1"] = c.shift(1)
    feats["high_1"] = h.shift(1)
    feats["low_1"] = l.shift(1)
    feats["vol_1"] = v.shift(1)

    # 2. Moving Averages & MA Ratios (M5 Horizons: 8, 21, 50, 200)
    ma8 = c.rolling(8).mean().shift(1)
    ma21 = c.rolling(21).mean().shift(1)
    ma50 = c.rolling(50).mean().shift(1)
    ma200 = c.rolling(200).mean().shift(1)

    feats["ma_ratio_8_21"] = (ma8 / (ma21 + 1e-9)) - 1.0
    feats["ma_ratio_8_50"] = (ma8 / (ma50 + 1e-9)) - 1.0
    feats["ma_ratio_21_50"] = (ma21 / (ma50 + 1e-9)) - 1.0
    feats["ma_ratio_50_200"] = (ma50 / (ma200 + 1e-9)) - 1.0

    # Distance to MAs in pips
    feats["dist_ma8_pips"] = (c.shift(1) - ma8) / pip
    feats["dist_ma21_pips"] = (c.shift(1) - ma21) / pip
    feats["dist_ma50_pips"] = (c.shift(1) - ma50) / pip

    # 3. Rolling Volatility & Volatility Ratios
    std8 = c.rolling(8).std().shift(1)
    std21 = c.rolling(21).std().shift(1)
    std50 = c.rolling(50).std().shift(1)

    feats["std_ratio_8_21"] = std8 / (std21 + 1e-9)
    feats["std_ratio_21_50"] = std21 / (std50 + 1e-9)
    feats["volatility_pips"] = std21 / pip

    # 4. Multi-period Returns
    feats["ret_1"] = (c.shift(1) - c.shift(2)) / (c.shift(2) + 1e-9)
    feats["ret_5"] = (c.shift(1) - c.shift(6)) / (c.shift(6) + 1e-9)
    feats["ret_21"] = (c.shift(1) - c.shift(22)) / (c.shift(22) + 1e-9)
    feats["ret_50"] = (c.shift(1) - c.shift(51)) / (c.shift(51) + 1e-9)

    # 5. Volume Dynamics
    vol_ma21 = v.rolling(21).mean().shift(1)
    feats["vol_ratio"] = v.shift(1) / (vol_ma21 + 1e-9)

    # 6. Microstructure Candle Shape
    rng = (h.shift(1) - l.shift(1)).clip(lower=1e-6)
    feats["body_ratio"] = (c.shift(1) - o.shift(1)) / rng
    feats["upper_wick"] = (h.shift(1) - np.maximum(c.shift(1), o.shift(1))) / rng
    feats["lower_wick"] = (np.minimum(c.shift(1), o.shift(1)) - l.shift(1)) / rng

    # 7. Targets
    # Target 1: Regresi drift harga 5 candle ke depan (dalam pips)
    feats["target_drift_pips"] = (c.shift(-5) - c) / pip
    
    # Target 2: Klasifikasi arah (1: NAIK >= +2.5 pip, 0: TURUN / FLAT)
    feats["target_direction"] = np.where(feats["target_drift_pips"] > 2.0, 1, 
                               np.where(feats["target_drift_pips"] < -2.0, -1, 0))

    # Actual Close price for notebook comparison
    feats["current_close"] = c
    feats["future_close"] = c.shift(-5)

    return feats.dropna()


def train_eurusd_predictive_model() -> dict:
    print("=" * 65)
    print("  PELATIHAN MODEL PREDIKTIF EURUSD (BERBASIS NOTEBOOK)")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File {DATA_PATH} tidak ditemukan.")

    df = pd.read_csv(DATA_PATH)
    print(f"Data Mentah : {len(df):,} bar M5 EURUSD")

    # Generate Features
    print("Mengekstraksi 20+ fitur multi-horizon...")
    feat_df = generate_eurusd_features(df)
    print(f"Data Fitur  : {len(feat_df):,} bar siap latih")

    # 90% Train / 10% Test Split (Chronological)
    n = len(feat_df)
    train_n = int(n * 0.90)
    test_n = n - train_n

    feature_cols = [col for col in feat_df.columns if col not in [
        "target_drift_pips", "target_direction", "current_close", "future_close"
    ]]

    X_train = feat_df[feature_cols].iloc[:train_n].values
    y_train_drift = feat_df["target_drift_pips"].iloc[:train_n].values
    y_train_dir = feat_df["target_direction"].iloc[:train_n].values

    X_test = feat_df[feature_cols].iloc[train_n:].values
    y_test_drift = feat_df["target_drift_pips"].iloc[train_n:].values
    y_test_dir = feat_df["target_direction"].iloc[train_n:].values

    print(f"Data Latih (90%): {train_n:,} sampel")
    print(f"Data Uji   (10%): {test_n:,} sampel (Out-of-sample)")

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Bagging Regressor (Notebook's Best Model Architecture)
    print("\n[1] Melatih Bagging Regressor (Ensemble Linear Regressors)...")
    base_lin = Ridge(alpha=10.0)
    bgr = BaggingRegressor(estimator=base_lin, n_estimators=50, random_state=42, n_jobs=-1)
    bgr.fit(X_train_scaled, y_train_drift)

    pred_drift = bgr.predict(X_test_scaled)

    rmse = math.sqrt(mean_squared_error(y_test_drift, pred_drift))
    mae = mean_absolute_error(y_test_drift, pred_drift)
    r2 = r2_score(y_test_drift, pred_drift)

    print(f"  • RMSE Drift : {rmse:.2f} pips")
    print(f"  • MAE Drift  : {mae:.2f} pips")
    print(f"  • R^2 Score  : {r2:.3f}")

    # 2. Evaluasi Sinyal Trading & Arah Out-of-Sample
    # Jika model memprediksi drift > +1.5 pip -> BUY. Jika < -1.5 pip -> SELL.
    buy_signals = pred_drift > 1.5
    sell_signals = pred_drift < -1.5

    buy_correct = (buy_signals & (y_test_drift > 0)).sum()
    buy_total = buy_signals.sum()
    buy_acc = (buy_correct / buy_total * 100.0) if buy_total > 0 else 0.0

    sell_correct = (sell_signals & (y_test_drift < 0)).sum()
    sell_total = sell_signals.sum()
    sell_acc = (sell_correct / sell_total * 100.0) if sell_total > 0 else 0.0

    total_trades = buy_total + sell_total
    total_correct = buy_correct + sell_correct
    overall_acc = (total_correct / total_trades * 100.0) if total_trades > 0 else 0.0

    print(f"\n[2] EVALUASI AKURASI ARAH (OUT-OF-SAMPLE TEST):")
    print(f"  • Sinyal BUY  : {buy_total} trade | Benar: {buy_correct} ({buy_acc:.1f}% Akurasi)")
    print(f"  • Sinyal SELL : {sell_total} trade | Benar: {sell_correct} ({sell_acc:.1f}% Akurasi)")
    print(f"  • Akurasi Keseluruhan: {overall_acc:.1f}% ({total_correct}/{total_trades} trade)")

    # 3. Simpan Hasil Model ke JSON
    # Ambil bobot rata-rata dari estimator ensemble
    avg_coef = np.mean([est.coef_ for est in bgr.estimators_], axis=0)
    avg_intercept = np.mean([est.intercept_ for est in bgr.estimators_])

    model_state = {
        "symbol": "EURUSD",
        "timeframe": "M5",
        "model_type": "BaggingRegressor_Ridge_Ensemble",
        "features": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": avg_coef.tolist(),
        "intercept": float(avg_intercept),
        "metrics": {
            "samples_train": train_n,
            "samples_test": test_n,
            "rmse_pips": round(rmse, 2),
            "mae_pips": round(mae, 2),
            "r2_score": round(r2, 3),
            "directional_accuracy_pct": round(overall_acc, 1),
            "buy_accuracy_pct": round(buy_acc, 1),
            "sell_accuracy_pct": round(sell_acc, 1),
            "total_test_signals": int(total_trades),
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OUTPUT_STATE, "w", encoding="utf-8") as f:
        json.dump(model_state, f, indent=2)

    print(f"\n[3] Model EURUSD berhasil disimpan ke: {OUTPUT_STATE}")
    return model_state


if __name__ == "__main__":
    train_eurusd_predictive_model()
