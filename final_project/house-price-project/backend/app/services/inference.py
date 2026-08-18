"""Inference service: loads the ML pipeline and runs predictions.

The model is loaded ONCE at application startup (via FastAPI lifespan) and
stored in a module-level variable so each request does NOT incur disk I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Module-level state — populated during startup
_pipeline: Any = None


def load_pipeline(model_path: str | Path) -> None:
    """Load the joblib pipeline from disk into the module-level variable.

    Call this once during application startup.

    Args:
        model_path: Path to the house_price.pkl file.
    """
    global _pipeline
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path.resolve()}'. "
            "Run the Jupyter notebook first to train and export the model."
        )
    _pipeline = joblib.load(path)
    logger.info("ML pipeline loaded from '%s'", path.resolve())


import numpy as np


def predict(df: pd.DataFrame) -> float:
    """Run inference on a single-row DataFrame.

    The scikit-learn Pipeline handles all preprocessing internally (imputation,
    scaling, one-hot encoding), so df must only contain the raw feature columns.

    Args:
        df: One-row pandas DataFrame with training-compatible column names.

    Returns:
        Predicted house price in INR (float).

    Raises:
        RuntimeError: If the pipeline has not been loaded yet.
    """
    if _pipeline is None:
        raise RuntimeError("Model pipeline is not loaded. Call load_pipeline() first.")

    prediction = _pipeline.predict(df)
    raw_val = float(prediction[0])
    # If the model outputs log-space predictions (e.g. < 30), invert to INR
    if raw_val < 30.0:
        price = float(np.expm1(raw_val))
    else:
        price = raw_val
    logger.info("Prediction: ₹%.2f", price)
    return price
