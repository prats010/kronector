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
    crashed_in_quali: bool = Field(
        False, description="Manually flag if the driver crashed in qualifying"
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
        ..., ge=0, le=100, description="Win probability as a percentage (0-100)"
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


class CompareRequest(BaseModel):
    """Request for Head-to-Head Driver Comparison."""

    driver1: str = Field(
        ...,
        examples=["Max Verstappen", "VER"],
        description="First driver to compare (name or 3-letter code)"
    )
    driver2: str = Field(
        ...,
        examples=["Kimi Antonelli", "ANT"],
        description="Second driver to compare (name or 3-letter code)"
    )
    season: Optional[int] = Field(None, description="Optional year (defaults to latest available)")
    round: Optional[int] = Field(None, description="Optional round number (defaults to latest available)")


class CompareResponse(BaseModel):
    """Head-to-Head Comparison Response."""

    driver1: PredictionMetadata
    driver2: PredictionMetadata
    driver1_win_probability: float
    driver2_win_probability: float
    shap_deltas: dict[str, float] = Field(
        ..., 
        description="Mathematical difference between driver 1 and driver 2 SHAP values (positive means driver 1 has the edge, negative means driver 2 has the edge)"
    )
    llm_analysis: str = Field(..., description="Groq's Head-to-Head tale of the tape")
