"""
KRONECTOR - Prediction Agent.

Takes the output from DataAgent (a DataFrame) and returns a typed prediction dictionary.
"""

from __future__ import annotations

import os
from typing import TypedDict

import mlflow.lightgbm
import pandas as pd
from mlflow.artifacts import download_artifacts

from ml.feature_engineering import load_encoders
from ml.predict import predict_dataframe


class PredictionOutput(TypedDict):
    probability: float
    shap_values: dict[str, float]
    feature_names: list[str]
    model_version: str
    run_id: str
    driver_name: str


def prediction_agent(dataframe: pd.DataFrame) -> PredictionOutput:
    """Run inference on a DataFrame representing an F1 driver query."""
    if dataframe.empty:
        raise ValueError("Cannot predict on an empty DataFrame")

    registered_model_name = "kronector-f1-lgbm"
    run_id = os.getenv("KRONECTOR_RUN_ID")

    if not run_id:
        raise RuntimeError("KRONECTOR_RUN_ID environment variable is required")

    # Load model from registry
    model_uri = f"models:/{registered_model_name}/latest"
    model = mlflow.lightgbm.load_model(model_uri)

    # Load encoders using run_id
    encoder_path = download_artifacts(
        run_id=run_id, artifact_path="encoders/label_encoders.pkl"
    )
    encoders = load_encoders(encoder_path)

    # Predict
    predictions = predict_dataframe(dataframe, model, encoders, explain=True)

    row = predictions.iloc[0]
    shap_values = dict(row.get("shap_values", {}))

    return {
        "probability": float(row["win_probability"]),
        "shap_values": shap_values,
        "feature_names": list(shap_values.keys()),
        "model_version": "latest",
        "run_id": run_id,
    }
