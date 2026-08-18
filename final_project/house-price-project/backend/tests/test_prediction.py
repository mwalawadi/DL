"""Pytest tests for the House Price Prediction API.

Run with:
    cd backend
    pytest tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Patch the lifespan so we don't need a real .pkl during testing
# ---------------------------------------------------------------------------
SAMPLE_LOCATIONS = {"Whitefield", "Koramangala", "HSR Layout", "Other"}

VALID_PAYLOAD = {
    "location": "Whitefield",
    "carpet_area_sqft": 1200.0,
    "floor_num": 3,
    "bathroom": 2,
    "balcony": 1,
    "furnishing": "Semi-Furnished",
    "transaction": "New Property",
    "ownership": "Freehold",
    "facing": "East",
}


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a mocked ML pipeline."""
    # Mock inference module so we never need a real .pkl
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = [4_250_000.0]

    with (
        patch("app.services.inference._pipeline", mock_pipeline),
        patch("app.services.preprocessing.load_locations", return_value=SAMPLE_LOCATIONS),
        patch("app.services.inference.load_pipeline", return_value=None),
    ):
        from app.main import app
        from app.api.routes.prediction import set_valid_locations

        set_valid_locations(SAMPLE_LOCATIONS)

        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Test 1 — Health check
# ---------------------------------------------------------------------------
def test_health(client):
    """GET /health should return 200 with {status: ok}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Test 2 — Happy path: valid prediction
# ---------------------------------------------------------------------------
def test_predict_valid(client):
    """POST /predict with valid payload should return 200 and a numeric price."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], (int, float))
    assert data["predicted_price"] > 0


# ---------------------------------------------------------------------------
# Test 3 — Invalid input: missing required field → 422
# ---------------------------------------------------------------------------
def test_predict_missing_field(client):
    """POST /predict with missing 'location' should return 422."""
    bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "location"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 4 — Invalid input: negative carpet area → 422
# ---------------------------------------------------------------------------
def test_predict_negative_area(client):
    """POST /predict with carpet_area_sqft <= 0 should return 422."""
    bad_payload = {**VALID_PAYLOAD, "carpet_area_sqft": -100.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 5 — Unknown location is handled gracefully (mapped to Other)
# ---------------------------------------------------------------------------
def test_predict_unknown_location(client):
    """POST /predict with unknown location should still return 200."""
    payload = {**VALID_PAYLOAD, "location": "UnknownCityXYZ"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
