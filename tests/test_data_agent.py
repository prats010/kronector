"""
Tests for agents.data_agent.

Run: python -m pytest tests/test_data_agent.py -v
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from agents.data_agent import (
    data_agent,
    parse_query_with_groq,
)
from ml.feature_engineering import prepare_model_data


@pytest.fixture
def agent_dataset(tmp_path):
    df = pd.DataFrame(
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
    path = tmp_path / "races.parquet"
    df.to_parquet(path, index=False)
    return path


def test_data_agent_returns_prediction_compatible_dataframe(agent_dataset):
    def parser(query):
        return {
            "season": 2023,
            "round": 1,
            "driver_id": "VER",
            "driver_name": "Max Verstappen",
        }

    result = data_agent("Will Verstappen win Bahrain 2023?", agent_dataset, parser)

    assert result["intent"]["driver_id"] == "VER"
    assert len(result["rows"]) == 1
    assert result["dataframe"].iloc[0]["driver_id"] == "VER"

    bundle, _ = prepare_model_data(result["dataframe"])
    assert len(bundle.X) == 1


def test_data_agent_returns_all_race_rows_when_no_driver(agent_dataset):
    def parser(query):
        return {"season": 2023, "round": 1, "driver_id": None, "driver_name": None}

    result = data_agent("Predict Bahrain 2023", agent_dataset, parser)

    assert len(result["dataframe"]) == 2


def test_data_agent_raises_for_missing_race(agent_dataset):
    def parser(query):
        return {"season": 2024, "round": 99, "driver_id": None, "driver_name": None}

    with pytest.raises(ValueError, match="No rows found"):
        data_agent("Predict a missing race", agent_dataset, parser)


def test_parse_query_with_groq_parses_json(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"season": 2023, "round": 1, '
                                '"driver_id": "ver", "driver_name": "Max Verstappen"}'
                            )
                        )
                    )
                ]
            )

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=FakeCompletions()
            )

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("agents.data_agent.Groq", FakeGroq)

    intent = parse_query_with_groq("Will Max win Bahrain 2023?")

    assert intent == {
        "season": 2023,
        "round": 1,
        "driver_id": "VER",
        "driver_name": "Max Verstappen",
    }
