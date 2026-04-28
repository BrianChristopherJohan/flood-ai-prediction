"""
Train XGBoost flood risk classifier on synthetic Sarawak weather data.

Generates 15,000 samples that reflect the bimodal Sarawak monsoon pattern:
  - Northeast Monsoon (Oct-Feb, DOY ~274-365/1-60): high rainfall, elevated flood risk
  - Southwest Monsoon (May-Sep, DOY ~121-273): moderate rainfall
  - Inter-monsoon transitions (Mar-Apr, Oct-Nov): variable, short-burst rainfall

Run from the project root:
    python scripts/train.py
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "flood_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

FEATURES = [
    'rain_1day', 'elevation', 'slope', 'u10', 'v10', 't2m', 'sp', 'tp',
    'ro', 'swvl1', 'rain_3day', 'rain_5day', 'rain_7day', 'rain_avg',
    'wind_speed', 'storm_intensity', 'slope_runoff_potential',
]

N_SAMPLES = 15_000
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _monsoon_intensity(doy: np.ndarray) -> np.ndarray:
    """
    Return monsoon intensity [0, 1] for each day-of-year.

    Peak of Northeast Monsoon is modelled around DOY 30 (Jan 30).
    Southwest Monsoon contributes a secondary peak around DOY 210 (Jul 29).
    """
    ne_monsoon = 0.5 + 0.5 * np.sin(2 * np.pi * (doy - 30) / 365) ** 2
    sw_monsoon = 0.25 * np.sin(2 * np.pi * (doy - 210) / 365) ** 2
    return np.clip(ne_monsoon + sw_monsoon, 0.0, 1.0)


def generate_dataset(n: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic meteorological dataset for Sarawak lowland river basins."""
    rng = np.random.RandomState(seed)
    logger.info("Generating %d synthetic samples (seed=%d)…", n, seed)

    doy = rng.randint(1, 366, size=n)
    monsoon = _monsoon_intensity(doy)

    # --- Rainfall ---
    base_rain = monsoon * 35.0 + rng.exponential(scale=7.0, size=n)
    rain_1day = np.maximum(0.0, base_rain + rng.normal(0, 6, size=n))
    rain_3day = np.maximum(0.0, base_rain * 2.6 + rng.normal(0, 10, size=n))
    rain_5day = np.maximum(0.0, base_rain * 4.2 + rng.normal(0, 16, size=n))
    rain_7day = np.maximum(0.0, base_rain * 5.8 + rng.normal(0, 20, size=n))
    rain_avg = np.maximum(0.0, base_rain * 0.92 + rng.normal(0, 2.5, size=n))

    # --- Terrain (Sarawak coastal lowlands: 0-50 m, gentle slopes) ---
    elevation = np.clip(rng.gamma(shape=2.0, scale=8.0, size=n), 0.5, 80.0)
    slope = np.clip(rng.gamma(shape=1.5, scale=2.5, size=n), 0.1, 30.0)
    slope_runoff_potential = np.clip(
        slope / 30.0 * 0.6 + monsoon * 0.3 + rng.uniform(0, 0.1, size=n), 0.0, 1.0
    )

    # --- ERA5-style atmospheric variables ---
    u10 = rng.normal(0, 3.0, size=n)
    v10 = rng.normal(0, 3.0, size=n)
    wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)
    t2m = 27.5 - monsoon * 1.8 + rng.normal(0, 0.6, size=n)
    sp = 101325.0 + rng.normal(0, 350, size=n)
    tp = np.maximum(0.0, base_rain * 0.82 + rng.normal(0, 3.5, size=n))
    ro = np.maximum(0.0, base_rain * 0.32 + rng.exponential(2.5, size=n))
    swvl1 = np.clip(monsoon * 0.42 + rng.normal(0, 0.05, size=n), 0.0, 0.55)

    # --- Storm events (rare but impactful) ---
    storm_intensity = np.clip(
        monsoon * 0.68 + rng.exponential(0.08, size=n), 0.0, 1.0
    )

    df = pd.DataFrame({
        'rain_1day': rain_1day,
        'elevation': elevation,
        'slope': slope,
        'u10': u10,
        'v10': v10,
        't2m': t2m,
        'sp': sp,
        'tp': tp,
        'ro': ro,
        'swvl1': swvl1,
        'rain_3day': rain_3day,
        'rain_5day': rain_5day,
        'rain_7day': rain_7day,
        'rain_avg': rain_avg,
        'wind_speed': wind_speed,
        'storm_intensity': storm_intensity,
        'slope_runoff_potential': slope_runoff_potential,
        'doy': doy,
        'monsoon': monsoon,
    })

    # --- Flood label (binary: 1 = flood event) ---
    # Physical rule: high cumulative rain + flat terrain + saturated soil → flood
    flood_score = (
        0.30 * (rain_1day / 80.0)
        + 0.25 * (rain_3day / 200.0)
        + 0.15 * (rain_7day / 400.0)
        + 0.10 * (storm_intensity)
        + 0.10 * (swvl1 / 0.55)
        + 0.05 * (ro / 40.0)
        + 0.05 * (slope_runoff_potential)
        - 0.10 * np.log1p(elevation) / np.log1p(80)
        - 0.05 * (slope / 30.0)
    )
    noise = rng.normal(0, 0.06, size=n)
    flood_prob = 1 / (1 + np.exp(-8.0 * (flood_score + noise - 0.38)))
    flood_label = (rng.uniform(size=n) < flood_prob).astype(int)

    df['flood'] = flood_label
    logger.info(
        "Label distribution — flood: %d (%.1f%%)  no-flood: %d (%.1f%%)",
        int(flood_label.sum()),
        100.0 * flood_label.mean(),
        int((1 - flood_label).sum()),
        100.0 * (1 - flood_label).mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> tuple[XGBClassifier, StandardScaler]:
    X = df[FEATURES].values
    y = df['flood'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pos_weight = float((y_train == 0).sum()) / max(1, float((y_train == 1).sum()))
    logger.info("XGBoost scale_pos_weight = %.2f", pos_weight)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("Training XGBoost classifier…")
    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )

    # --- Evaluation ---
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    logger.info("=" * 60)
    logger.info("  Accuracy : %.4f", acc)
    logger.info("  ROC-AUC  : %.4f", auc)
    logger.info("=" * 60)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))

    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion matrix:\n%s", cm)

    # --- Feature importances ---
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logger.info("Top feature importances:")
    for rank, idx in enumerate(sorted_idx, 1):
        logger.info("  %2d. %-30s %.4f", rank, FEATURES[idx], importances[idx])

    return model, scaler


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(model: XGBClassifier, scaler: StandardScaler) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", MODEL_PATH)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved → %s", SCALER_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Flood AI — training pipeline starting")
    logger.info("Project root: %s", ROOT)

    df = generate_dataset(n=N_SAMPLES, seed=RANDOM_STATE)
    model, scaler = train(df)
    save_artifacts(model, scaler)

    logger.info("Training complete. Run the API with:")
    logger.info("  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
