"""Pydantic schemas for prediction request and response."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for the /predict endpoint.

    All feature names must exactly match those used during model training.
    """

    location: str = Field(..., description="Property location / neighbourhood")
    carpet_area_sqft: float = Field(..., gt=0, description="Carpet area in square feet")
    floor_num: int = Field(..., ge=0, description="Floor number (0 = ground floor)")
    bathroom: int = Field(..., ge=0, description="Number of bathrooms")
    balcony: int = Field(..., ge=0, description="Number of balconies")
    bhk: int | None = Field(default=None, ge=1, description="Number of bedrooms (BHK)")
    furnishing: str = Field(..., description="Furnishing status (e.g. Furnished, Semi-Furnished, Unfurnished)")
    transaction: str = Field(..., description="Transaction type (e.g. New Property, Resale)")
    ownership: str = Field(..., description="Ownership type (e.g. Freehold, Leasehold)")
    facing: str = Field(..., description="Direction the property faces (e.g. East, North)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for the /predict endpoint."""

    predicted_price: float = Field(..., description="Predicted house price in INR")
