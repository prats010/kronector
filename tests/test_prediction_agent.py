"""
Tests for PredictionAgent.

Run: python -m pytest tests/test_prediction_agent.py -v
"""

import pickle
import sys
from types import SimpleNamespace

import pytest

from agents.prediction_agent import prediction_agent
from ml.feature_engineering import prepare_model_data
from tests.test_feature_engineering import sample_race_df  # noqa: F401
from tests.test_predict import DummyExplainer, DummyModel


def test_prediction_agent(sample_race_df, monkeypatch, tmp_path):
    monkeypatch.setenv("KRONECTOR_RUN_ID", "dummy_run_id")

    # Mock mlflow.lightgbm.load_model
    import mlflow.lightgbm
    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda uri: DummyModel())

    # Mock download_artifacts and save some dummy encoders there
    _, encoders = prepare_model_data(sample_race_df)
    
    encoder_path = tmp_path / "label_encoders.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(encoders, f)
        
    monkeypatch.setattr(
        "agents.prediction_agent.download_artifacts",
        lambda run_id, artifact_path: str(encoder_path)
    )

    # Mock shap
    dummy_shap = SimpleNamespace(TreeExplainer=DummyExplainer)
    monkeypatch.setitem(sys.modules, "shap", dummy_shap)

    # Call agent
    result = prediction_agent(sample_race_df.iloc[:1])

    assert result["run_id"] == "dummy_run_id"
    assert result["model_version"] == "latest"
    assert result["probability"] == 0.25  # DummyModel returns [0.75, 0.25]
    assert isinstance(result["shap_values"], dict)
    assert len(result["feature_names"]) == len(result["shap_values"])
    assert list(result["shap_values"].keys()) == result["feature_names"]


def test_prediction_agent_missing_env_var(sample_race_df, monkeypatch):
    monkeypatch.delenv("KRONECTOR_RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="KRONECTOR_RUN_ID"):
        prediction_agent(sample_race_df)


def test_prediction_agent_empty_dataframe(monkeypatch):
    import pandas as pd
    monkeypatch.setenv("KRONECTOR_RUN_ID", "dummy_run_id")
    with pytest.raises(ValueError, match="empty DataFrame"):
        prediction_agent(pd.DataFrame())
