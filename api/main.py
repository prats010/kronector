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
from agents.critique_agent import critique_agent
from agents.synthesis_agent import synthesis_agent
from agents.compare_agent import compare_agent
from api.schemas import (
    DriverInfo,
    ErrorResponse,
    HealthResponse,
    PredictionMetadata,
    PredictionRequest,
    PredictionResponse,
    RaceInfo,
    CompareRequest,
    CompareResponse
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
_prerace_data: dict[str, pd.DataFrame] = {}  # keyed by "season_round" e.g. "2026_7"


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
        
        # Allow MLflow to use the local filesystem store (required in MLflow 2.13+)
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        
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

        # Load any pre-race parquet files from data_output/prerace/
        prerace_dir = Path("data_output/prerace")
        if prerace_dir.exists():
            for pf in prerace_dir.glob("*.parquet"):
                try:
                    prerace_df = pd.read_parquet(pf)
                    if not prerace_df.empty:
                        season = int(prerace_df["season"].iloc[0])
                        round_num = int(prerace_df["round"].iloc[0])
                        key = f"{season}_{round_num}"
                        _prerace_data[key] = prerace_df
                        circuit = prerace_df["circuit_id"].iloc[0]
                        logger.info(f"✓ Pre-race data loaded: {key} ({circuit}) — {len(prerace_df)} drivers")
                except Exception as e:
                    logger.warning(f"Could not load pre-race file {pf}: {e}")

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

    if _races_data is None and not _prerace_data:
        raise HTTPException(
            status_code=503,
            detail="Race data not loaded. Check data_output/fastf1_races.parquet",
        )

    try:
        # Parse query → intent using the main historical parquet
        # If the round isn't in the main parquet, fall back to pre-race data
        from agents.data_agent import parse_query_with_groq, _filter_by_intent
        intent = parse_query_with_groq(request.query)

        df = pd.DataFrame()
        is_prerace = False

        # 1. Try the main historical parquet first
        if _races_data is not None:
            try:
                df = _filter_by_intent(_races_data, intent, filter_driver=False)
            except (ValueError, KeyError):
                df = pd.DataFrame()

        # 2. If not found in historical data, try pre-race data
        if df.empty and _prerace_data:
            logger.info(f"Round not in historical data — searching pre-race store for intent: {intent}")
            for key, prerace_df in _prerace_data.items():
                try:
                    df = _filter_by_intent(prerace_df, intent, filter_driver=False)
                    if not df.empty:
                        is_prerace = True
                        logger.info(f"✓ Found match in pre-race data: {key}")
                        break
                except (ValueError, KeyError):
                    continue

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No matching data for query. Season={intent.get('season')}, "
                    f"GP='{intent.get('grand_prix')}'. "
                    "If this is an upcoming race, run: "
                    "python -m scripts.build_prerace_rows --season <year> --round <num>"
                ),
            )

        # Generate predictions
        predictions = predict_dataframe(df, _model, _encoders, explain=True)
        
        # Get top 3 contenders for context before filtering
        top_df = predictions.sort_values(by="win_probability", ascending=False).head(3)
        top_contenders = ", ".join(
            [f"{r['driver_name']} ({r['win_probability']*100:.1f}%)" for _, r in top_df.iterrows()]
        )
        
        # Now filter down to the requested driver if specified
        driver_id = intent.get("driver_id")
        driver_name = intent.get("driver_name")
        
        import re
        if driver_id:
            d_id = str(driver_id).strip().upper()
            predictions = predictions[predictions["driver_id"].astype(str).str.upper() == d_id]
        elif driver_name:
            d_name = str(driver_name).strip().lower()
            predictions = predictions[
                predictions["driver_name"].astype(str).str.lower().str.contains(re.escape(d_name), na=False)
            ]
            
        if predictions.empty:
            raise HTTPException(status_code=400, detail=f"No predictions matched driver intent: {intent}")
        
        # If the user asked "Who will win?" and matched multiple drivers,
        # sort by probability to find the most likely winner!
        if len(predictions) > 1:
            predictions = predictions.sort_values(by="win_probability", ascending=False)
            
        pred_row = predictions.iloc[0]
        
        # grid_position is stripped from metadata by predict_dataframe — look it up
        # directly from the source df using the matched driver_id.
        matched_driver_id = str(pred_row["driver_id"])
        grid_pos_series = df.loc[
            df["driver_id"].astype(str) == matched_driver_id, "grid_position"
        ]
        grid_pos = float(grid_pos_series.iloc[0]) if not grid_pos_series.empty else 0.0

        # 1. Format raw prediction output
        raw_prob = float(pred_row["win_probability"])
        shap_values = pred_row.get("shap_values", {})
        
        prediction_output = {
            "probability": raw_prob,
            "shap_values": dict(shap_values),
            "feature_names": list(shap_values.keys()) if shap_values else [],
            "model_version": "latest",
            "run_id": "api",
            "driver_name": str(pred_row["driver_name"]),
            "is_prerace": is_prerace,
            "quali_status": "Crash" if getattr(request, "crashed_in_quali", False) else str(pred_row.get("quali_status", "Finished")),
        }
        
        # 2. Mathematical Critique
        critique = critique_agent(prediction_output)
        
        # 3. LLM Synthesis
        synthesis = synthesis_agent(request.query, prediction_output, critique, top_contenders=top_contenders)

        return PredictionResponse(
            win_probability=round(raw_prob * 100, 2),
            metadata=PredictionMetadata(
                season=int(pred_row["season"]),
                round=int(pred_row["round"]),
                driver_id=str(pred_row["driver_id"]),
                driver_name=str(pred_row["driver_name"]),
                team=str(pred_row["team"]),
                grid_position=grid_pos,
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


@app.post(
    "/predict/compare",
    response_model=CompareResponse,
    description="Compare two drivers head-to-head using mathematical SHAP Deltas",
    tags=["Prediction"]
)
def compare_drivers(request: CompareRequest):
    """Head-to-head driver matchup endpoint."""
    if _model is None or _encoders is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Set KRONECTOR_MODEL_RUN_ID."
        )

    try:
        import re
        import json
        is_prerace = False
        df = pd.DataFrame()
        race_name = "Unknown Race"

        # 1. Season and round specified?
        if request.season and request.round:
            prerace_file = Path(f"data_output/prerace/prerace_*_{request.season}.parquet")
            import glob
            files = glob.glob(str(prerace_file))
            
            found = False
            for f in files:
                pdf = pd.read_parquet(f)
                if not pdf.empty and int(pdf["season"].iloc[0]) == request.season and int(pdf["round"].iloc[0]) == request.round:
                    df = pdf
                    is_prerace = True
                    race_name = f"{request.season} Round {request.round}"
                    found = True
                    break
            
            if not found:
                global _races_data
                if _races_data is not None and not _races_data.empty:
                    df = _races_data[
                        (_races_data["season"] == request.season) & 
                        (_races_data["round"] == request.round)
                    ]
                    race_name = f"{request.season} Round {request.round}"
        
        # 2. No season/round specified, use latest
        if df.empty:
            prerace_dir = Path("data_output/prerace")
            if prerace_dir.exists():
                files = list(prerace_dir.glob("*.parquet"))
                if files:
                    df = pd.read_parquet(files[0])
                    is_prerace = True
                    s = int(df["season"].iloc[0])
                    r = int(df["round"].iloc[0])
                    race_name = f"{s} Round {r}"
            
            if df.empty and _races_data is not None and not _races_data.empty:
                latest_season = _races_data["season"].max()
                latest_round = _races_data[_races_data["season"] == latest_season]["round"].max()
                df = _races_data[
                    (_races_data["season"] == latest_season) & 
                    (_races_data["round"] == latest_round)
                ]
                race_name = f"{latest_season} Round {latest_round}"

        if df.empty:
            raise HTTPException(status_code=400, detail="Could not find race data for comparison.")

        predictions = predict_dataframe(df, _model, _encoders, explain=True)

        d1_str = request.driver1.strip().lower()
        d1_df = predictions[
            (predictions["driver_id"].str.lower() == d1_str) | 
            (predictions["driver_name"].str.lower().str.contains(re.escape(d1_str), na=False))
        ]
        if d1_df.empty:
            raise HTTPException(status_code=400, detail=f"Driver '{request.driver1}' not found in race {race_name}.")
        d1_row = d1_df.iloc[0]

        d2_str = request.driver2.strip().lower()
        d2_df = predictions[
            (predictions["driver_id"].str.lower() == d2_str) | 
            (predictions["driver_name"].str.lower().str.contains(re.escape(d2_str), na=False))
        ]
        if d2_df.empty:
            raise HTTPException(status_code=400, detail=f"Driver '{request.driver2}' not found in race {race_name}.")
        d2_row = d2_df.iloc[0]

        d1_prob = float(d1_row["win_probability"]) * 100
        d2_prob = float(d2_row["win_probability"]) * 100

        d1_shap = d1_row["shap_values"]
        d2_shap = d2_row["shap_values"]
        if isinstance(d1_shap, str):
            d1_shap = json.loads(d1_shap)
        if isinstance(d2_shap, str):
            d2_shap = json.loads(d2_shap)
        
        deltas = {}
        for feature in d1_shap.keys():
            if feature in d2_shap:
                deltas[feature] = d1_shap[feature] - d2_shap[feature]

        llm_analysis = compare_agent(
            driver1_name=str(d1_row["driver_name"]),
            driver1_prob=d1_prob,
            driver1_status=str(d1_row.get("quali_status", "Finished")),
            driver2_name=str(d2_row["driver_name"]),
            driver2_prob=d2_prob,
            driver2_status=str(d2_row.get("quali_status", "Finished")),
            shap_deltas=deltas,
            race_context=f"{race_name} {'(Pre-Race)' if is_prerace else '(Historical)'}"
        )

        return CompareResponse(
            driver1=PredictionMetadata(
                season=int(d1_row["season"]),
                round=int(d1_row["round"]),
                driver_id=str(d1_row["driver_id"]),
                driver_name=str(d1_row["driver_name"]),
                team=str(d1_row["team"]),
                grid_position=float(d1_row.get("grid_position", 0.0)),
            ),
            driver2=PredictionMetadata(
                season=int(d2_row["season"]),
                round=int(d2_row["round"]),
                driver_id=str(d2_row["driver_id"]),
                driver_name=str(d2_row["driver_name"]),
                team=str(d2_row["team"]),
                grid_position=float(d2_row.get("grid_position", 0.0)),
            ),
            driver1_win_probability=round(d1_prob, 2),
            driver2_win_probability=round(d2_prob, 2),
            shap_deltas=deltas,
            llm_analysis=llm_analysis
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")


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

@app.post(
    "/predict/compare",
    response_model=CompareResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query or driver not found"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_compare(request: CompareRequest) -> CompareResponse:
    """
    Head-to-Head Driver Comparison.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set KRONECTOR_MODEL_RUN_ID.",
        )

    if _races_data is None and not _prerace_data:
        raise HTTPException(
            status_code=503,
            detail="Race data not loaded. Check data_output/fastf1_races.parquet",
        )

    season = request.season
    round_num = request.round

    # If not provided, find the latest pre-race data or historical data
    if not season or not round_num:
        if _prerace_data:
            latest_key = list(_prerace_data.keys())[-1]
            season, round_num = map(int, latest_key.split("_"))
        elif _races_data is not None:
            season = int(_races_data["season"].max())
            round_num = int(_races_data[_races_data["season"] == season]["round"].max())

    df = pd.DataFrame()
    key = f"{season}_{round_num}"
    if _prerace_data and key in _prerace_data:
        df = _prerace_data[key].copy()
    elif _races_data is not None:
        df = _races_data[(_races_data["season"] == season) & (_races_data["round"] == round_num)].copy()

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No data for season {season} round {round_num}"
        )

    # Find driver 1 and driver 2
    def find_driver(driver_str: str, data: pd.DataFrame):
        driver_str = driver_str.lower()
        match = data[data["driver_id"].str.lower() == driver_str]
        if match.empty:
            match = data[data["driver_name"].str.lower().str.contains(driver_str)]
        return match

    d1_df = find_driver(request.driver1, df)
    d2_df = find_driver(request.driver2, df)

    if d1_df.empty:
        raise HTTPException(status_code=400, detail=f"Driver 1 '{request.driver1}' not found in race")
    if d2_df.empty:
        raise HTTPException(status_code=400, detail=f"Driver 2 '{request.driver2}' not found in race")

    d1_row = d1_df.iloc[[0]]
    d2_row = d2_df.iloc[[0]]

    # Generate predictions
    from ml.predict import predict_dataframe
    d1_pred = predict_dataframe(d1_row, _model, _encoders, explain=True)
    d2_pred = predict_dataframe(d2_row, _model, _encoders, explain=True)

    d1_prob = float(d1_pred["win_probability"].iloc[0])
    d2_prob = float(d2_pred["win_probability"].iloc[0])

    d1_shap = d1_pred["shap_values"].iloc[0]
    d2_shap = d2_pred["shap_values"].iloc[0]

    # Calculate SHAP deltas (D1 - D2)
    shap_deltas = {}
    for feature in d1_shap.keys():
        shap_deltas[feature] = float(d1_shap[feature] - d2_shap.get(feature, 0.0))

    d1_name = str(d1_row["driver_name"].iloc[0])
    d2_name = str(d2_row["driver_name"].iloc[0])
    
    # Get quali status if available
    d1_status = str(d1_row.get("quali_status", pd.Series(["Unknown"])).iloc[0]) if "quali_status" in d1_row.columns else "Unknown"
    d2_status = str(d2_row.get("quali_status", pd.Series(["Unknown"])).iloc[0]) if "quali_status" in d2_row.columns else "Unknown"

    circuit = str(df["circuit_id"].iloc[0])
    race_context = f"{season} Round {round_num} at {circuit}"

    from agents.compare_agent import compare_agent
    llm_analysis = compare_agent(
        driver1_name=d1_name,
        driver1_prob=d1_prob * 100,
        driver1_status=d1_status,
        driver2_name=d2_name,
        driver2_prob=d2_prob * 100,
        driver2_status=d2_status,
        shap_deltas=shap_deltas,
        race_context=race_context
    )

    return CompareResponse(
        driver1=PredictionMetadata(
            season=season, round=round_num,
            driver_id=str(d1_row["driver_id"].iloc[0]),
            driver_name=d1_name,
            team=str(d1_row["team"].iloc[0]),
            grid_position=float(d1_row["grid_position"].iloc[0])
        ),
        driver2=PredictionMetadata(
            season=season, round=round_num,
            driver_id=str(d2_row["driver_id"].iloc[0]),
            driver_name=d2_name,
            team=str(d2_row["team"].iloc[0]),
            grid_position=float(d2_row["grid_position"].iloc[0])
        ),
        driver1_win_probability=d1_prob,
        driver2_win_probability=d2_prob,
        shap_deltas=shap_deltas,
        llm_analysis=llm_analysis
    )

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
