"""
sync_models.py — Copy pre-trained models from FYP-RainfallView and retrain
the primary XGBoost model + scaler for flood-ai-prediction.

Usage (run from project root):
    python scripts/sync_models.py [--skip-train]

Steps:
  1. Locate FYP-RainfallView/models/ relative to the workspace root.
  2. Copy flood_model_xgc_v2.pkl, flood_model_lgbmc.pkl, flood_model_cboost.pkl
     into this service's models/ directory.
  3. (Unless --skip-train) Re-run the training pipeline to regenerate
     flood_model.pkl + scaler.pkl with the latest synthetic Sarawak dataset.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).resolve().parent
ROOT       = THIS_DIR.parent                    # flood-ai-prediction/
WORKSPACE  = ROOT.parent                        # fyp_2026/
FYP_SRC    = WORKSPACE / "FYP-RainfallView" / "models"
MODELS_DST = ROOT / "models"

# Source → destination filename mapping
COPY_MAP: dict[str, str] = {
    "flood_model_xgc_v2.pkl": "flood_model_xgc_v2.pkl",
    "flood_model_lgbmc.pkl":  "flood_model_lgbmc.pkl",
    "flood_model_cboost.pkl": "flood_model_cboost.pkl",
}


def sync_fyp_models() -> None:
    """Copy FYP-RainfallView model artefacts into models/."""
    if not FYP_SRC.exists():
        logger.error(
            "FYP-RainfallView models not found at %s. "
            "Ensure the FYP-RainfallView folder is at the same workspace level.",
            FYP_SRC,
        )
        sys.exit(1)

    MODELS_DST.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    for src_name, dst_name in COPY_MAP.items():
        src = FYP_SRC / src_name
        dst = MODELS_DST / dst_name

        if not src.exists():
            logger.warning("Source not found, skipping: %s", src)
            skipped += 1
            continue

        shutil.copy2(src, dst)
        size_kb = dst.stat().st_size / 1024
        logger.info("Copied %-40s → models/%s  (%.1f KB)", src_name, dst_name, size_kb)
        copied += 1

    logger.info("Sync complete — copied: %d, skipped: %d", copied, skipped)


def run_training() -> None:
    """Run train.py to regenerate flood_model.pkl + scaler.pkl."""
    logger.info("Running training pipeline (scripts/train.py)…")
    result = subprocess.run(
        [sys.executable, str(THIS_DIR / "train.py")],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        logger.error("Training failed with exit code %d", result.returncode)
        sys.exit(result.returncode)
    logger.info("Training pipeline finished successfully.")


def verify_models() -> None:
    """Quick sanity check — load each .pkl and confirm predict_proba exists."""
    import pickle

    logger.info("Verifying model artefacts…")
    ok = 0
    fail = 0
    for pkl in sorted(MODELS_DST.glob("*.pkl")):
        try:
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            has_pp = hasattr(obj, "predict_proba") or hasattr(obj, "transform")
            status = "OK" if has_pp else "WARN (no predict_proba)"
            logger.info("  %-45s %s", pkl.name, status)
            ok += 1
        except Exception as exc:
            logger.error("  %-45s FAILED — %s", pkl.name, exc)
            fail += 1

    logger.info("Verification — passed: %d, failed: %d", ok, fail)
    if fail:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-train", action="store_true", help="Skip re-training; only sync FYP models.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Flood AI — model sync & train pipeline")
    logger.info("  Workspace : %s", WORKSPACE)
    logger.info("  FYP source: %s", FYP_SRC)
    logger.info("  Models dst: %s", MODELS_DST)
    logger.info("=" * 60)

    sync_fyp_models()

    if not args.skip_train:
        run_training()

    verify_models()

    logger.info("All done. Start the API with:")
    logger.info("  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
