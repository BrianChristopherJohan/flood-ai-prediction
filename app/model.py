"""
Flood-AI Prediction — model registry and inference engine.

Model hierarchy (loaded in priority order):
  1. flood_model.pkl         — freshly retrained XGBoost (Sarawak synthetic, train.py)
  2. flood_model_xgc_v2.pkl  — XGBoost v2 from FYP-RainfallView
  3. flood_model_lgbmc.pkl   — LightGBM from FYP-RainfallView   (optional)
  4. flood_model_cboost.pkl  — CatBoost from FYP-RainfallView   (optional)

When more than one model is loaded, predictions use soft-voting ensemble averaging.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature contract ──────────────────────────────────────────────────────────
FEATURES: list[str] = [
    "rain_1day", "elevation", "slope", "u10", "v10", "t2m", "sp", "tp",
    "ro", "swvl1", "rain_3day", "rain_5day", "rain_7day", "rain_avg",
    "wind_speed", "storm_intensity", "slope_runoff_potential",
]

# ── Model registry entries ────────────────────────────────────────────────────
_REGISTRY: dict[str, dict] = {
    "xgb-retrained": {
        "path": Path("models/flood_model.pkl"),
        "label": "XGBoost (Sarawak retrained)",
        "source": "scripts/train.py — 15 000 synthetic samples, Sarawak monsoon",
        "weight": 2.0,   # higher weight in ensemble — calibrated with scaler
    },
    "xgb-v2": {
        "path": Path("models/flood_model_xgc_v2.pkl"),
        "label": "XGBoost v2 (FYP-RainfallView)",
        "source": "FYP-RainfallView original training",
        "weight": 1.0,
    },
    "lgbm": {
        "path": Path("models/flood_model_lgbmc.pkl"),
        "label": "LightGBM (FYP-RainfallView)",
        "source": "FYP-RainfallView original training",
        "weight": 1.0,
    },
    "catboost": {
        "path": Path("models/flood_model_cboost.pkl"),
        "label": "CatBoost (FYP-RainfallView)",
        "source": "FYP-RainfallView original training",
        "weight": 1.0,
    },
}

# Runtime state
_loaded_models: dict[str, object] = {}    # key → fitted estimator
_scaler: Optional[object] = None
_scaler_path = Path("models/scaler.pkl")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model() -> None:
    """Load all available models from the registry and the scaler."""
    global _scaler

    # Scaler (fitted on training data by scripts/train.py)
    if _scaler_path.exists():
        with open(_scaler_path, "rb") as f:
            _scaler = pickle.load(f)
        logger.info("Scaler loaded from %s", _scaler_path)
    else:
        logger.warning(
            "scaler.pkl not found — run scripts/train.py to generate it. "
            "Falling back to unscaled inference."
        )

    # Models
    for name, meta in _REGISTRY.items():
        p: Path = meta["path"]
        if not p.exists():
            logger.debug("Model '%s' not found at %s — skipping.", name, p)
            continue
        try:
            with open(p, "rb") as f:
                clf = pickle.load(f)
            _loaded_models[name] = clf
            logger.info("Loaded %-18s ← %s", f"'{name}'", p.name)
        except Exception as exc:
            logger.warning("Failed to load '%s' from %s: %s", name, p, exc)

    if _loaded_models:
        logger.info(
            "%d model(s) active: %s",
            len(_loaded_models),
            ", ".join(_loaded_models.keys()),
        )
    else:
        logger.warning(
            "No models loaded — using rule-based fallback. "
            "Copy *.pkl files to models/ or run scripts/train.py."
        )


# ── Inference helpers ─────────────────────────────────────────────────────────

def _prepare_features(features_df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix; apply scaler when available."""
    X = features_df[FEATURES].values.astype(float)
    if _scaler is not None:
        X = _scaler.transform(X)
    return X


def _proba_from_model(clf: object, X: np.ndarray) -> np.ndarray:
    """Return (N,) probability array for the positive class."""
    try:
        return (clf.predict_proba(X)[:, 1] * 100.0).round(2)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("predict_proba failed for %s: %s", type(clf).__name__, exc)
        return np.full(X.shape[0], 50.0)


def _ensemble_proba(X: np.ndarray) -> np.ndarray:
    """Weighted soft-voting across all loaded models."""
    total_weight = 0.0
    weighted_sum = np.zeros(X.shape[0])

    for name, clf in _loaded_models.items():
        w = _REGISTRY[name]["weight"]
        probas = _proba_from_model(clf, X)
        weighted_sum += probas * w
        total_weight += w

    return weighted_sum / max(total_weight, 1e-9)


def _map_proba_to_level(proba: float) -> int:
    if proba < 30:
        return 0
    if proba < 50:
        return 1
    if proba < 75:
        return 2
    return 3


# ── Public API ────────────────────────────────────────────────────────────────

def predict_flood_risk(features_df: pd.DataFrame) -> list[dict]:
    """
    Return a list of dicts with keys 'level' (0–3) and 'probability' (0–100).

    Uses ensemble if multiple models are loaded; falls back to rule-based
    heuristic when no models are available.
    """
    if not _loaded_models:
        return _rule_based(features_df)

    X = _prepare_features(features_df)
    probas = _ensemble_proba(X)
    return [
        {"level": _map_proba_to_level(float(p)), "probability": float(p)}
        for p in probas
    ]


def _rule_based(df: pd.DataFrame) -> list[dict]:
    """Conservative physics-based fallback when no ML model is available."""
    results = []
    for _, row in df.iterrows():
        rain = float(row.get("rain_1day", 0)) + float(row.get("rain_3day", 0)) * 0.3
        proba = min(100.0, max(0.0, rain * 1.8))
        results.append({
            "level": _map_proba_to_level(proba),
            "probability": round(proba, 1),
        })
    return results


# ── Status queries ────────────────────────────────────────────────────────────

def is_model_loaded() -> bool:
    return bool(_loaded_models)


def get_registry_status() -> dict:
    """Return a serialisable summary of all models (loaded and missing)."""
    out: dict = {}
    for name, meta in _REGISTRY.items():
        loaded = name in _loaded_models
        out[name] = {
            "label": meta["label"],
            "source": meta["source"],
            "weight": meta["weight"],
            "file": meta["path"].name,
            "loaded": loaded,
            "file_exists": meta["path"].exists(),
            "size_kb": round(meta["path"].stat().st_size / 1024, 1) if meta["path"].exists() else None,
        }
    return out
