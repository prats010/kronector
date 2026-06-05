"""
Tests for prediction helpers.

Run: python -m pytest tests/test_predict.py -v
"""

import sys
from types import SimpleNamespace

import numpy as np

from ml.feature_engineering import prepare_model_data
from ml.predict import predict_dataframe
from tests.test_feature_engineering import sample_race_df  # noqa: F401


class DummyModel:
    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.75), np.full(len(X), 0.25)])


class DummyExplainer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, X):
        return np.ones((len(X), len(X.columns)))


def test_predict_dataframe_returns_shap_dicts(sample_race_df, monkeypatch):
    _, encoders = prepare_model_data(sample_race_df)
    dummy_shap = SimpleNamespace(TreeExplainer=DummyExplainer)
    monkeypatch.setitem(sys.modules, "shap", dummy_shap)

    predictions = predict_dataframe(
        sample_race_df.iloc[:2], DummyModel(), encoders, explain=True
    )

    assert "shap_values" in predictions.columns
    assert predictions["win_probability"].tolist() == [0.25, 0.25]
    assert isinstance(predictions.loc[0, "shap_values"], dict)
    assert set(predictions.loc[0, "shap_values"]) == set(
        prepare_model_data(sample_race_df.iloc[:2], encoders=encoders)[0].X.columns
    )


def test_predict_dataframe_can_skip_explanations(sample_race_df):
    _, encoders = prepare_model_data(sample_race_df)

    predictions = predict_dataframe(
        sample_race_df.iloc[:2], DummyModel(), encoders, explain=False
    )

    assert "shap_values" not in predictions.columns
