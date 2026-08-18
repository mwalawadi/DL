"""Preprocessing helpers: convert raw API input into a model-ready DataFrame.

The scikit-learn Pipeline inside house_price.pkl already handles:
  - SimpleImputer (median / most_frequent)
  - StandardScaler
  - OneHotEncoder

This module's only job is to build the one-row pandas DataFrame with
the EXACT column names used during training and to map unknown locations
to the "Other" bucket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.schemas.prediction import PredictionRequest
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_locations(path: str | Path) -> set[str]:
    """Load the allowed location values from locations.json.

    Args:
        path: Absolute or relative path to locations.json.

    Returns:
        Set of valid location strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data)


def build_dataframe(request: PredictionRequest, valid_locations: set[str]) -> pd.DataFrame:
    """Convert a PredictionRequest into a one-row DataFrame ready for the Pipeline.

    Unknown locations are mapped to "Other" to match the training preprocessing.

    Args:
        request: Validated request object from the API.
        valid_locations: Set of location values seen during training.

    Returns:
        Single-row pandas DataFrame with training-compatible column names.
    """
    location_grouped = request.location if request.location in valid_locations else "Other"
    if request.location not in valid_locations:
        logger.warning("Unknown location '%s' → mapped to 'Other'", request.location)

    bhk_val = float(request.bhk) if request.bhk is not None else float(max(1, request.bathroom))
    row = {
        "carpet_area_sqft": request.carpet_area_sqft,
        "floor_num": float(request.floor_num),
        "bathroom": float(request.bathroom),
        "balcony": float(request.balcony),
        "bhk": bhk_val,
        "location_grouped": location_grouped,
        "Furnishing": request.furnishing,
        "Transaction": request.transaction,
        "Ownership": request.ownership,
        "facing": request.facing,
    }

    return pd.DataFrame([row])
