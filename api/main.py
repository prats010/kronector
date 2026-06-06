"""
KRONECTOR API — FastAPI application.

Endpoints:
  POST /predict/f1 — Predict win probability from natural language query
  GET /drivers — List all drivers
  GET /races/{season} — List races for a season
  GET /health — System health check

Run:
    python -m uvicorn api.main:app --reload
    
Visit:
    http://localhost:8000/docs (interactive API docs)
    http://localhost:8000/redoc (ReDoc)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from agents.data_agent import data_agent, QueryIntent
from api.schemas import (
    DriverInfo,
    ErrorResponse,
    HealthResponse,
    PredictionMetadata,
    PredictionRequest,
    PredictionResponse,
    RaceInfo,
)
from ml.predict import load_model_and_encoders, predict_dataframe


logger = logging.getLogger(__name__)


# ============================================================================
# Global state
# ============================================================================

_model = None
_encoders = None
_races_data: pd.DataFrame | None = None
_drivers_set: set[str] = set()


def patch_mlruns_paths():
    """Locate all meta.yaml files in mlruns/ and dynamically update their absolute paths
    to point to the current active workspace directory. This ensures MLflow runs can be
    loaded correctly on different hosts (e.g. locally or in Hugging Face Spaces).
    """
    import re
    from pathlib import Path

    current_dir = Path(__file__).resolve().parent.parent
    mlruns_dir = current_dir / "mlruns"
    if not mlruns_dir.exists():
        logger.warning("mlruns directory not found, skipping path patching")
        return
        
    new_base_uri = mlruns_dir.resolve().as_uri()
    logger.info(f"Dynamically patching mlruns URIs to: {new_base_uri}")
    
    pattern = re.compile(r"file:///[^\n]*?/mlruns")
    
    patched_count = 0
    for meta_path in mlruns_dir.rglob("meta.yaml"):
        try:
            content = meta_path.read_text(encoding="utf-8")
            new_content = pattern.sub(new_base_uri, content)
            if new_content != content:
                meta_path.write_text(new_content, encoding="utf-8")
                patched_count += 1
        except Exception as e:
            logger.error(f"Failed to patch {meta_path}: {e}")
            
    if patched_count > 0:
        logger.info(f"Successfully patched {patched_count} meta.yaml files with current workspace path.")


# ============================================================================
# Lifespan events
# ============================================================================


async def lifespan(app: FastAPI):
    """Load model and data on startup."""
    global _model, _encoders, _races_data, _drivers_set

    logger.info("Loading KRONECTOR model and data...")
    
    try:
        # Load model (requires MLflow run_id)
        import os
        from dotenv import load_dotenv
        
        load_dotenv()  # Load variables from .env file
        
        # Ensure MLFLOW_TRACKING_URI is set correctly to local mlruns if empty or unset
        if not os.getenv("MLFLOW_TRACKING_URI"):
            os.environ["MLFLOW_TRACKING_URI"] = "./mlruns"
            logger.info("Setting MLFLOW_TRACKING_URI to default ./mlruns")
            
        # Dynamically patch meta.yaml files to handle environment transitions
        try:
            patch_mlruns_paths()
        except Exception as e:
            logger.warning(f"Error during mlruns path patching: {e}")

        run_id = os.getenv("KRONECTOR_MODEL_RUN_ID")
        if run_id:
            try:
                _model, _encoders = load_model_and_encoders(run_id)
                logger.info(f"✓ Model loaded from run {run_id}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        else:
            logger.warning("KRONECTOR_MODEL_RUN_ID not set; predictions unavailable")

        # Load race data
        data_path = Path("data_output/fastf1_races.parquet")
        if data_path.exists():
            _races_data = pd.read_parquet(data_path)
            _drivers_set = set(_races_data["driver_id"].unique())
            logger.info(
                f"✓ Data loaded: {len(_races_data)} rows, "
                f"{len(_drivers_set)} drivers"
            )
        else:
            logger.warning(f"Data not found: {data_path}")

    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield
    logger.info("KRONECTOR API shutdown")


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(
    title="KRONECTOR F1 Intelligence API",
    description="Predict F1 race outcomes using natural language queries",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — configurable via CORS_ORIGINS env var (comma-separated).
# Default allows local development; set CORS_ORIGINS in production.
import os

_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8501,http://localhost:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================


from agents.critique_agent import critique_agent
from agents.synthesis_agent import synthesis_agent

@app.post(
    "/predict/f1",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_f1(request: PredictionRequest) -> PredictionResponse:
    """
    Predict F1 race win probability from natural language query.

    Example:
        ```
        POST /predict/f1
        {
            "query": "What's Max Verstappen's win probability at Monaco 2023?"
        }
        ```

    Returns:
        - `win_probability`: Float 0.0-1.0
        - `metadata`: Race/driver/team info
        - `shap_values`: Feature importance dict
        - `llm_explanation`: Natural language response
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set KRONECTOR_MODEL_RUN_ID.",
        )

    if _races_data is None:
        raise HTTPException(
            status_code=503,
            detail="Race data not loaded. Check data_output/fastf1_races.parquet",
        )

    try:
        # Parse query → intent
        result = data_agent(request.query)
        intent = result["intent"]
        df = result["dataframe"]

        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No matching data for query. "
                f"Season {intent['season']}, round {intent['round']}",
            )

        # Generate predictions
        predictions = predict_dataframe(df, _model, _encoders, explain=True)
        
        # If the user asked "Who will win?" and matched multiple drivers,
        # sort by probability to find the most likely winner!
        if len(predictions) > 1:
            predictions = predictions.sort_values(by="win_probability", ascending=False)
            
        pred_row = predictions.iloc[0]
        
        # 1. Format raw prediction output
        raw_prob = float(pred_row["win_probability"])
        shap_values = pred_row.get("shap_values", {})
        
        prediction_output = {
            "probability": raw_prob,
            "shap_values": dict(shap_values),
            "feature_names": list(shap_values.keys()) if shap_values else [],
            "model_version": "latest",
            "run_id": "api",
            "driver_name": str(pred_row["driver_name"])
        }
        
        # 2. Mathematical Critique
        critique = critique_agent(prediction_output)
        
        # 3. LLM Synthesis
        synthesis = synthesis_agent(request.query, prediction_output, critique)

        return PredictionResponse(
            win_probability=raw_prob,
            metadata=PredictionMetadata(
                season=int(pred_row["season"]),
                round=int(pred_row["round"]),
                driver_id=str(pred_row["driver_id"]),
                driver_name=str(pred_row["driver_name"]),
                team=str(pred_row["team"]),
                grid_position=float(pred_row.get("grid_position", 0.0)),  # Default if missing
            ),
            shap_values=shap_values,
            llm_explanation=synthesis["final_response"],
            confidence_rating=critique["confidence_rating"]
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal prediction error"
        ) from e


@app.get("/drivers", response_model=list[DriverInfo])
async def list_drivers(
    season: Optional[int] = Query(None, description="Filter by season"),
) -> list[DriverInfo]:
    """
    List all available drivers.

    Query Parameters:
        - `season` (optional): Filter to drivers in a specific season

    Returns:
        List of driver IDs and names
    """
    if _races_data is None:
        raise HTTPException(
            status_code=503, detail="Race data not loaded"
        )

    df = _races_data
    if season:
        df = df[df["season"] == season]

    if len(df) == 0:
        return []

    drivers = (
        df[["driver_id", "driver_name", "team"]]
        .drop_duplicates()
        .sort_values("driver_id")
    )

    return [
        DriverInfo(
            driver_id=row["driver_id"],
            driver_name=row["driver_name"],
            team=row["team"],
        )
        for _, row in drivers.iterrows()
    ]


@app.get("/races/{season}", response_model=list[RaceInfo])
async def list_races(season: int) -> list[RaceInfo]:
    """
    List all races in a season.

    Path Parameters:
        - `season`: F1 season year (e.g., 2023)

    Returns:
        List of races with round number and circuit name
    """
    if _races_data is None:
        raise HTTPException(
            status_code=503, detail="Race data not loaded"
        )

    races = (
        _races_data[_races_data["season"] == season][
            ["round", "circuit_id"]
        ]
        .drop_duplicates()
        .sort_values("round")
    )

    if len(races) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No races found for season {season}",
        )

    return [
        RaceInfo(
            season=season,
            round=int(row["round"]),
            name=str(row["circuit_id"]),
        )
        for _, row in races.iterrows()
    ]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    System health check.

    Returns:
        - `status`: "healthy" or "degraded"
        - `model_loaded`: Whether model is available
        - `data_available`: Whether race data is available
    """
    status = "healthy"
    if _model is None or _races_data is None:
        status = "degraded"

    return HealthResponse(
        status=status,
        model_loaded=_model is not None,
        data_available=_races_data is not None,
    )


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "KRONECTOR F1 Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================================
# Error handlers
# ============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return {
        "detail": str(exc),
        "error_code": "VALIDATION_ERROR",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
