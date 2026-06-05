"""
KRONECTOR - LightGBM training entry point.

Training flow:
  - load race parquet data
  - prepare features and fit categorical LabelEncoders
  - evaluate all TimeSeriesSplit folds and log averaged CV metrics
  - retrain a final LightGBM model on the full dataset
  - log the model, fitted encoders, and SHAP summary to MLflow
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ml.feature_engineering import (
    create_time_series_splits,
    prepare_model_data,
    save_encoders,
)


REGISTERED_MODEL_NAME = "kronector-f1-lgbm"


LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "class_weight": "balanced",
    "verbosity": -1,
}


def _positive_class_shap_values(shap_values):
    """Normalize SHAP binary-class outputs to one 2D array."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


def _build_model():
    import lightgbm as lgb

    return lgb.LGBMClassifier(**LIGHTGBM_PARAMS)


def _cross_validate(bundle, n_splits: int) -> dict[str, float]:
    """Evaluate all time-series folds and return averaged metrics."""
    fold_metrics = []

    for fold, (train_idx, valid_idx) in enumerate(
        create_time_series_splits(bundle.X, n_splits=n_splits), start=1
    ):
        model = _build_model()
        X_train = bundle.X.iloc[train_idx]
        y_train = bundle.y.iloc[train_idx]
        X_valid = bundle.X.iloc[valid_idx]
        y_valid = bundle.y.iloc[valid_idx]

        model.fit(X_train, y_train)
        valid_prob = model.predict_proba(X_valid)[:, 1]

        metrics = {
            "fold": fold,
            "log_loss": log_loss(y_valid, valid_prob, labels=[0, 1]),
        }
        if y_valid.nunique() > 1:
            metrics["roc_auc"] = roc_auc_score(y_valid, valid_prob)
        fold_metrics.append(metrics)

    metric_names = sorted(
        metric for metrics in fold_metrics for metric in metrics if metric != "fold"
    )
    averaged = {}
    for metric in metric_names:
        values = [fold[metric] for fold in fold_metrics if metric in fold]
        averaged[f"cv_mean_{metric}"] = float(np.mean(values))

    return averaged


def _save_shap_summary(model, X: pd.DataFrame, path: Path) -> None:
    """Create a SHAP summary plot for the final fitted model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = _positive_class_shap_values(explainer.shap_values(X))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def train_model(
    data_path: str = "data_output/fastf1_races.parquet",
    experiment_name: str = "kronector-week3",
    n_splits: int = 5,
) -> str:
    """
    Train a LightGBM model and log model artifacts to MLflow.

    Returns:
        MLflow run id.
    """
    import mlflow
    import mlflow.lightgbm

    df = pd.read_parquet(data_path)
    bundle, encoders = prepare_model_data(df)

    cv_metrics = _cross_validate(bundle, n_splits=n_splits)

    final_model = _build_model()
    final_model.fit(bundle.X, bundle.y)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(final_model.get_params())
        mlflow.log_params(
            {
                "n_splits": n_splits,
                "n_features": len(bundle.feature_columns),
                "n_rows": len(bundle.X),
                "model_type": "LightGBM",
            }
        )
        mlflow.log_metrics(cv_metrics)
        mlflow.lightgbm.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            encoder_path = tmp_path / "label_encoders.pkl"
            save_encoders(encoders, str(encoder_path))
            mlflow.log_artifact(str(encoder_path), artifact_path="encoders")

            shap_path = tmp_path / "shap_summary.png"
            _save_shap_summary(final_model, bundle.X, shap_path)
            mlflow.log_artifact(str(shap_path), artifact_path="explainability")

        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KRONECTOR LightGBM model")
    parser.add_argument("--data-path", default="data_output/fastf1_races.parquet")
    parser.add_argument("--experiment-name", default="kronector-week3")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    run_id = train_model(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        n_splits=args.n_splits,
    )
    print(f"MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
