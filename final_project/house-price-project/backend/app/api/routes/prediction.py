"""API routes for house price prediction."""

from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services import inference, preprocessing
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Loaded once during startup via lifespan (see main.py)
_valid_locations: set[str] = set()


def set_valid_locations(locations: set[str]) -> None:
    """Inject the set of valid locations into the route module.

    Args:
        locations: Set of location strings from locations.json.
    """
    global _valid_locations
    _valid_locations = locations


@router.get("/health", summary="Health check", tags=["Health"])
async def health_check() -> dict:
    """Return the API health status.

    Returns:
        JSON ``{"status": "ok"}`` when the service is running.
    """
    return {"status": "ok"}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict house price",
    tags=["Prediction"],
)
async def predict_price(request: PredictionRequest) -> PredictionResponse:
    """Predict the price of a house given its features.

    The request body is validated by Pydantic.  The exported scikit-learn
    Pipeline handles all preprocessing (imputation, scaling, encoding)
    internally — we only build the feature DataFrame here.

    Args:
        request: Validated prediction request.

    Returns:
        JSON with ``predicted_price`` in INR.

    Raises:
        HTTPException 500: If inference fails unexpectedly.
    """
    try:
        df = preprocessing.build_dataframe(request, _valid_locations)
        price = inference.predict(df)
        return PredictionResponse(predicted_price=price)
    except RuntimeError as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
