"""
Integration tests for api.main — FastAPI endpoints.

Run:
    python -m pytest tests/test_api_endpoints.py -v -s
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from agents.data_agent import QueryIntent


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_startup(monkeypatch):
    """Mock model and data loading."""
    import pandas as pd

    # Create minimal mock data
    mock_data = pd.DataFrame(
        {
            "season": [2023, 2023],
            "round": [1, 1],
            "driver_id": ["VER", "HAM"],
            "driver_name": ["Max Verstappen", "Lewis Hamilton"],
            "team": ["Red Bull Racing", "Mercedes"],
            "grid_position": [1.0, 2.0],
            "finish_position": [1.0, 2.0],
            "circuit_id": ["Bahrain Grand Prix", "Bahrain Grand Prix"],
            "sector_1_time": [28.7, 28.9],
            "sector_2_time": [38.5, 38.7],
            "sector_3_time": [22.4, 22.6],
            "avg_lap_time_practice": [95.1, 95.4],
            "tire_compound": [0.0, 1.0],
            "tire_age_laps": [14.0, 12.0],
            "fresh_tire": [0.0, 1.0],
            "pit_stop_count": [2.0, 2.0],
            "team_pit_speed": [2.5, 2.8],
            "weather_temp_track": [31.0, 31.0],
            "weather_rainfall": [0.0, 0.0],
            "telemetry_available": [True, True],
        }
    )

    # Mock data loading
    import api.main

    api.main._races_data = mock_data
    api.main._drivers_set = {"VER", "HAM"}

    # Mock model
    class MockModel:
        def predict_proba(self, X):
            import numpy as np

            return np.array([[0.3, 0.7], [0.6, 0.4]])

    api.main._model = MockModel()
    api.main._encoders = {}

    yield

    # Cleanup
    api.main._races_data = None
    api.main._drivers_set = set()
    api.main._model = None
    api.main._encoders = None


def test_health_endpoint(client):
    """GET /health returns status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "data_available" in data


def test_root_endpoint(client):
    """GET / returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "KRONECTOR F1 Intelligence API"
    assert "docs" in data


def test_predict_requires_model(client, mock_startup):
    """POST /predict/f1 returns 503 if model not loaded."""
    import api.main

    api.main._model = None
    response = client.post(
        "/predict/f1", json={"query": "Verstappen 2023 Bahrain"}
    )
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_predict_requires_query_length(client, mock_startup):
    """POST /predict/f1 validates query length."""
    response = client.post("/predict/f1", json={"query": "ab"})  # Too short
    assert response.status_code == 422  # Validation error


def test_predict_invalid_query(client, mock_startup):
    """POST /predict/f1 handles invalid queries gracefully."""

    def mock_parser(q: str) -> QueryIntent:
        # Return impossible race
        return {"season": 2099, "round": 999}

    import api.main

    api.main.data_agent = lambda q: (__import__("agents.data_agent", fromlist=["data_agent"]).data_agent(q, parser=mock_parser))

    response = client.post(
        "/predict/f1", json={"query": "Impossible race"}
    )
    # May be 400 (validation) or 500 (internal error)
    assert response.status_code in [400, 500]


def test_list_drivers(client, mock_startup):
    """GET /drivers returns driver list."""
    response = client.get("/drivers")
    assert response.status_code == 200
    drivers = response.json()
    assert len(drivers) == 2
    assert any(d["driver_id"] == "VER" for d in drivers)
    assert any(d["driver_id"] == "HAM" for d in drivers)


def test_list_drivers_by_season(client, mock_startup):
    """GET /drivers?season=2023 filters drivers."""
    response = client.get("/drivers?season=2023")
    assert response.status_code == 200
    drivers = response.json()
    assert len(drivers) == 2


def test_list_drivers_nonexistent_season(client, mock_startup):
    """GET /drivers?season=2099 returns empty list."""
    response = client.get("/drivers?season=2099")
    assert response.status_code == 200
    drivers = response.json()
    assert len(drivers) == 0


def test_list_races(client, mock_startup):
    """GET /races/{season} returns race list."""
    response = client.get("/races/2023")
    assert response.status_code == 200
    races = response.json()
    assert len(races) == 1
    assert races[0]["round"] == 1
    assert races[0]["season"] == 2023


def test_list_races_nonexistent_season(client, mock_startup):
    """GET /races/{season} for nonexistent season returns 404."""
    response = client.get("/races/2099")
    assert response.status_code == 404
    assert "No races found" in response.json()["detail"]


def test_predict_with_mock_encoders(client, mock_startup):
    """POST /predict/f1 handles predictions when encoders are available."""
    import api.main
    from unittest.mock import patch
    import numpy as np

    # Create minimal encoders dict
    from sklearn.preprocessing import LabelEncoder

    encoders = {}
    for col in ["team", "track_type", "regulation_era"]:
        enc = LabelEncoder()
        enc.fit([f"{col}_1", f"{col}_2", "unknown"])
        encoders[col] = enc

    api.main._encoders = encoders

    # Create a mock that tracks calls
    with patch("api.main.predict_dataframe") as mock_predict:
        mock_predict.return_value = pd.DataFrame(
            {
                "season": [2023],
                "round": [1],
                "driver_id": ["VER"],
                "driver_name": ["Max Verstappen"],
                "team": ["Red Bull Racing"],
                "grid_position": [1.0],
                "finish_position": [1.0],
                "circuit_id": ["Bahrain"],
                "win_probability": [0.75],
                "shap_values": [{"grid_position": 0.3, "sector_1_time": 0.2}],
            }
        )

        with patch("api.main.data_agent") as mock_da:
            mock_da.return_value = {
                "query": "Verstappen 2023",
                "intent": {"season": 2023, "round": 1, "driver_id": "VER"},
                "rows": [],
                "dataframe": pd.DataFrame({"season": [2023]}),
            }

            response = client.post(
                "/predict/f1", json={"query": "Verstappen 2023 Bahrain"}
            )
            # Verify predict_dataframe was called
            assert mock_predict.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
