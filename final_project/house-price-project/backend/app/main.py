"""FastAPI application entry point.

Uses the lifespan context manager to load the ML pipeline ONCE at startup
so that each request does not incur disk I/O or model deserialization overhead.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.prediction import router as prediction_router, set_valid_locations
from app.core.config import settings
from app.services import inference
from app.services.preprocessing import load_locations
from app.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Resolve paths relative to the backend/ directory
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent.parent  # …/backend/
MODEL_PATH = BACKEND_DIR / settings.model_path
LOCATIONS_PATH = BACKEND_DIR / settings.locations_path


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Application lifespan: load model on startup, clean up on shutdown."""
    logger.info("Loading ML pipeline from '%s'", MODEL_PATH)
    inference.load_pipeline(MODEL_PATH)

    logger.info("Loading locations from '%s'", LOCATIONS_PATH)
    valid_locations = load_locations(LOCATIONS_PATH)
    set_valid_locations(valid_locations)
    logger.info("Loaded %d valid locations", len(valid_locations))

    yield  # Application is running

    logger.info("Shutting down House Price API")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Predicts Indian residential property prices using a trained "
        "scikit-learn Random Forest pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server and any configured frontend URL
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(prediction_router)
