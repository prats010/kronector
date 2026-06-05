"""
KRONECTOR API — Pydantic schemas for request/response validation.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Natural language query for F1 prediction."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["What's Max's win probability at Monaco 2023?"],
    )


class PredictionMetadata(BaseModel):
    """Metadata about the predicted race/driver."""

    season: int
    round: int
    driver_id: str
    driver_name: str
    team: str
    grid_position: float


class PredictionResponse(BaseModel):
    """Win probability prediction with SHAP explanations."""

    win_probability: float = Field(
        ..., ge=0, le=1, description="Probability of winning (0.0-1.0)"
    )
    metadata: PredictionMetadata
    shap_values: Optional[dict[str, float]] = Field(
        None, description="SHAP feature importance dict"
    )
    llm_explanation: Optional[str] = Field(
        None, description="Natural language explanation from Llama3"
    )
    confidence_rating: Optional[str] = Field(
        None, description="CritiqueAgent's confidence evaluation"
    )


class DriverInfo(BaseModel):
    """Driver summary info."""

    driver_id: str
    driver_name: str
    team: Optional[str] = None


class RaceInfo(BaseModel):
    """Race summary info."""

    season: int
    round: int
    name: str  # Circuit name


class HealthResponse(BaseModel):
    """System health status."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool
    data_available: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: str = Field(..., examples=["INVALID_QUERY", "MODEL_NOT_LOADED"])
