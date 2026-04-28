# flood-ai-prediction

![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-ff6600)
![License](https://img.shields.io/badge/license-MIT-green)

**Multi-model ensemble flood risk prediction microservice — 4-level risk assessments for Sarawak / Sabah, Malaysia.**

## Overview

`flood-ai-prediction` is a FastAPI microservice that exposes flood risk predictions using a **weighted soft-voting ensemble** of four ML models sourced from two origins:

| Model key | Type | Source |
|---|---|---|
| `xgb-retrained` | XGBoost (weight ×2) | `scripts/train.py` — 15 000 synthetic Sarawak samples, fitted scaler |
| `xgb-v2` | XGBoost | FYP-RainfallView (BrianChristopherJohan) |
| `lgbm` | LightGBM | FYP-RainfallView |
| `catboost` | CatBoost | FYP-RainfallView |

All models share the same 17-feature input contract. When multiple models are loaded, their probability scores are merged via weighted averaging before thresholding into a risk level. If no model is available, the service falls back to a deterministic physics-based rule so the API never returns 503.

The service is consumed by `flood-website-crm` (Hourly / Daily / Weekly / Monthly charts) and `flood-mobile-community`.

*Inspired by: [FYP-RainfallView](https://github.com/BrianChristopherJohan) by BrianChristopherJohan.*

## Features

- **4-model soft-voting ensemble** — XGBoost (×2), LightGBM, CatBoost with configurable weights
- **17-feature input contract** — rainfall accumulations, ERA5-style atmospheric variables, terrain attributes, storm intensity
- **4-level risk output** — `0 Normal`, `1 Alert`, `2 Warning`, `3 Critical` mapped from probability thresholds
- **Hourly endpoint** — 24-hour risk breakdown for any ISO date
- **Daily endpoint** — full 365-day risk calendar grouped by month, including per-day hourly drill-down
- **Weekly endpoint** — Q1–Q4 quarterly risk summary (52 weeks)
- **Monthly endpoint** — 12-month aggregate risk with labels and average probabilities
- **Per-node prediction** — POST endpoint accepting real-time sensor readings (water level, rainfall, terrain) for a specific node ID
- **Rule-based fallback** — deterministic prediction when the XGBoost model file is absent, so the API never returns 503
- **Model info endpoint** — exposes feature list, probability thresholds, and model metadata
- **Swagger UI** — interactive API documentation at `/docs`; ReDoc at `/redoc`
- **CORS** — configurable allowed origins via environment variable

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| FastAPI | 0.111.0 | ASGI web framework |
| Uvicorn | 0.29.0 | ASGI server |
| XGBoost | 2.0.3 | Gradient-boosted tree classifier |
| scikit-learn | 1.4.2 | `StandardScaler`, metrics, train/test split |
| pandas | 2.2.2 | Feature DataFrame construction |
| numpy | 1.26.4 | Numerical operations |
| Pydantic | 2.7.1 | Request/response schema validation |
| python-dotenv | 1.0.1 | Environment variable loading |
| Python | 3.11 | Runtime |

## Architecture

```
flood-website-crm  (:3000)
        │  GET /api/v1/predict/daily?year=2026
        │  GET /api/v1/predict/hourly?date=2026-04-28
        │  POST /api/v1/predict/node
        │
        ▼
flood-ai-prediction  (:8000)
        │  XGBoost model (models/flood_model.pkl)
        │  StandardScaler (models/scaler.pkl)
        │
        ▼  Falls back to rule-based predictor if model not trained
```

The service is stateless — it loads the trained model once at startup and serves predictions from memory.

## Prerequisites

- **Python** 3.11 (3.10+ should work)
- **pip** ≥ 23.x (or use a virtual environment manager such as `venv` or `conda`)
- No external database or cache required — the model artifacts are local `.pkl` files

## Getting Started

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-org/floodwatch.git
cd floodwatch/flood-ai-prediction

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Sync models and run the training pipeline

This single command copies the pre-trained FYP-RainfallView models into `models/` and then re-runs the training pipeline to generate a properly calibrated `flood_model.pkl` + `scaler.pkl`.

```bash
python scripts/sync_models.py
```

Expected output:
```
Copied flood_model_xgc_v2.pkl  → models/flood_model_xgc_v2.pkl
Copied flood_model_lgbmc.pkl   → models/flood_model_lgbmc.pkl
Copied flood_model_cboost.pkl  → models/flood_model_cboost.pkl
...
Accuracy : 0.6430  |  ROC-AUC  : 0.6930
Model saved  → models/flood_model.pkl
Scaler saved → models/scaler.pkl
Verification — passed: 5, failed: 0
```

To skip re-training and only sync the FYP models:
```bash
python scripts/sync_models.py --skip-train
```

To retrain only (without syncing):
```bash
python scripts/train.py
```

> **Tip:** If you skip this step entirely, the service starts with the rule-based fallback. All endpoints remain functional.

> **Model files and `.gitignore`:** `.pkl` files are excluded from git (they are large binaries). Always run `sync_models.py` after a fresh clone.

### 4. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

The API is now available at [http://localhost:8000](http://localhost:8000).
Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Production start (no reload)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ALLOWED_ORIGINS` | Comma-separated CORS allowed origins | `*` |
| `PORT` | Port (used by Docker / process managers) | `8000` |

Example `.env`:

```env
ALLOWED_ORIGINS=http://localhost:3000,https://floodwatch.yourdomain.com
```

Load automatically via `python-dotenv` if a `.env` file is present in the project root.

## API Endpoints

### Health

| Method | Path | Description | Auth |
|---|---|---|---|
| `GET` | `/health` | Service health check; reports `model_loaded` status | Public |
| `GET` | `/` | Root — returns service info and links to `/docs` | Public |

### Predictions (`/api/v1/predict/...`)

| Method | Path | Query Params | Description |
|---|---|---|---|
| `GET` | `/api/v1/predict/hourly` | `date` (ISO, e.g. `2026-04-28`) | 24-hour risk breakdown for a specific date |
| `GET` | `/api/v1/predict/daily` | `year` (int, default 2026) | 365-day risk calendar grouped by month |
| `GET` | `/api/v1/predict/weekly` | `year` (int, default 2026) | Q1–Q4 quarterly risk (52 weekly averages) |
| `GET` | `/api/v1/predict/monthly` | `year` (int, default 2026) | 12-month aggregate risk with labels |
| `POST` | `/api/v1/predict/node` | — | Risk for a specific sensor node (body below) |
| `GET` | `/api/v1/model/info` | — | Model metadata, feature list, thresholds |

### `POST /api/v1/predict/node` — request body

```json
{
  "node_id": "102503180",
  "water_level": 1,
  "rain_1day": 45.0,
  "rain_3day": 110.0,
  "rain_5day": 175.0,
  "rain_7day": 230.0,
  "rain_avg": 38.0,
  "elevation": 12.0,
  "slope": 2.5,
  "wind_speed": 6.2,
  "storm_intensity": 0.35
}
```

All fields except `node_id` are optional and default to representative Sarawak lowland values.

### Risk Level Scale

| Level | Label | Probability |
|---|---|---|
| `0` | Normal | < 30 % |
| `1` | Alert | 30 – 50 % |
| `2` | Warning | 50 – 75 % |
| `3` | Critical | ≥ 75 % |

### Model Features (17 inputs)

| Feature | Description |
|---|---|
| `rain_1day` | 24-hour rainfall accumulation (mm) |
| `rain_3day` | 3-day rainfall accumulation (mm) |
| `rain_5day` | 5-day rainfall accumulation (mm) |
| `rain_7day` | 7-day rainfall accumulation (mm) |
| `rain_avg` | Rolling average daily rainfall (mm) |
| `elevation` | Terrain elevation above sea level (m) |
| `slope` | Terrain slope (degrees) |
| `slope_runoff_potential` | Composite slope × soil saturation index |
| `u10` | 10 m zonal (east-west) wind component (m/s) |
| `v10` | 10 m meridional (north-south) wind component (m/s) |
| `wind_speed` | Scalar wind speed (m/s) |
| `t2m` | 2 m air temperature (°C) |
| `sp` | Surface pressure (Pa) |
| `tp` | Total precipitation flux (mm) |
| `ro` | Surface runoff (mm) |
| `swvl1` | Volumetric soil water layer 1 (m³/m³) |
| `storm_intensity` | Composite storm intensity index [0, 1] |

## Project Structure

```
flood-ai-prediction/
├── app/
│   ├── main.py             # FastAPI app factory, middleware, router includes
│   ├── model.py            # Model loading, prediction logic, rule-based fallback
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/
│       ├── predict.py      # All /api/v1/predict/* and /api/v1/model/info endpoints
│       └── health.py       # GET /health
├── models/
│   ├── flood_model.pkl         # XGBoost retrained (generated by sync_models.py / train.py)
│   ├── scaler.pkl              # StandardScaler  (generated by sync_models.py / train.py)
│   ├── flood_model_xgc_v2.pkl  # XGBoost v2      (from FYP-RainfallView via sync_models.py)
│   ├── flood_model_lgbmc.pkl   # LightGBM        (from FYP-RainfallView via sync_models.py)
│   ├── flood_model_cboost.pkl  # CatBoost        (from FYP-RainfallView via sync_models.py)
│   └── .gitkeep
├── scripts/
│   ├── sync_models.py      # One-step: copy FYP models + retrain
│   └── train.py            # Standalone XGBoost training pipeline
├── requirements.txt
└── README.md
```

## Docker

```bash
# Build
docker build -t floodwatch-ai .

# Run
docker run -p 8000:8000 \
  -e ALLOWED_ORIGINS=http://localhost:3000 \
  floodwatch-ai
```

To run the full stack (recommended):

```bash
cd ../deploy
cp .env.example .env
docker compose up -d
```

> **Note:** The Docker image must include pre-trained model files (`models/flood_model.pkl` and `models/scaler.pkl`). Either run `python scripts/train.py` locally before building, or add a `RUN python scripts/train.py` step to the `Dockerfile`.

## Retraining the Model

The synthetic dataset uses a fixed random seed (`RANDOM_STATE = 42`) for reproducibility. To retrain with a different sample size or seed, edit the constants at the top of `scripts/train.py` and re-run:

```bash
python scripts/train.py
```

The `models/` directory will be updated in-place. Restart the running API to pick up the new model.

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feat/your-feature`
2. Commit your changes following [Conventional Commits](https://www.conventionalcommits.org/)
3. Push and open a Pull Request against `main`
4. Ensure the service starts cleanly (`uvicorn app.main:app --reload`) before requesting review

## License

This project is licensed under the [MIT License](../LICENSE).

---

Part of the **FloodWatch** flood monitoring system for Sarawak, Malaysia.
