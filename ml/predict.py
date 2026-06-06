"""
KRONECTOR - Prediction helpers.

Inference must reuse the LabelEncoders fitted during training. Do not call
prepare_model_data without the loaded encoders in production prediction paths.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ml.feature_engineering import load_encoders, prepare_model_data


def load_model_and_encoders(run_id: str):
    """Load a logged MLflow model and its fitted categorical encoders."""
    import mlflow.lightgbm
    from mlflow.artifacts import download_artifacts
    import logging
    
    logger = logging.getLogger(__name__)

    try:
        model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    except Exception as e:
        logger.warning(f"Could not load from runs:/ URI: {e}. Trying registered model...")
        try:
            model = mlflow.lightgbm.load_model("models:/kronector-f1-lgbm/latest")
        except Exception as e2:
            logger.error(f"Failed to load registered model: {e2}")
            raise
    encoder_path = download_artifacts(
        run_id=run_id, artifact_path="encoders/label_encoders.pkl"
    )
    encoders = load_encoders(encoder_path)
    return model, encoders


def _positive_class_shap_values(shap_values):
    """Normalize SHAP binary-class outputs to one 2D array."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


def _explain_predictions(model, X: pd.DataFrame) -> list[dict[str, float]]:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = _positive_class_shap_values(explainer.shap_values(X))
    return [
        {
            feature_name: float(shap_value)
            for feature_name, shap_value in zip(X.columns, row_values)
        }
        for row_values in shap_values
    ]


def predict_dataframe(
    df: pd.DataFrame, model, encoders: dict, explain: bool = True
) -> pd.DataFrame:
    """Return win probabilities and optional SHAP dictionaries."""
    bundle, _ = prepare_model_data(df, encoders=encoders)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(bundle.X)[:, 1]
    else:
        probabilities = model.predict(bundle.X)

    result = bundle.metadata.copy()
    result["win_probability"] = probabilities
    if explain:
        result["shap_values"] = _explain_predictions(model, bundle.X)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KRONECTOR model inference")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-path", default="data_output/fastf1_races.parquet")
    args = parser.parse_args()

    model, encoders = load_model_and_encoders(args.run_id)
    df = pd.read_parquet(args.data_path)
    predictions = predict_dataframe(df, model, encoders)
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()
