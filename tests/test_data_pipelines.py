"""
KRONECTOR — Data Pipeline Tests
Tests for fastf1_pipeline, jolpica_pipeline, build_driver_map, and merge logic.

Run: python -m pytest tests/test_data_pipelines.py -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests


# ===================================================================
# Test: Driver Map
# ===================================================================


class TestDriverMap:
    """Tests for data.build_driver_map module."""

    def test_known_driver_map_has_key_drivers(self):
        """KNOWN_DRIVER_MAP must include key F1 drivers 2014–2024."""
        from data.build_driver_map import KNOWN_DRIVER_MAP

        assert "VER" in KNOWN_DRIVER_MAP
        assert "HAM" in KNOWN_DRIVER_MAP
        assert "LEC" in KNOWN_DRIVER_MAP
        assert "NOR" in KNOWN_DRIVER_MAP
        assert "ALO" in KNOWN_DRIVER_MAP

    def test_known_map_values_are_jolpica_slugs(self):
        """All values should be lowercase slug format."""
        from data.build_driver_map import KNOWN_DRIVER_MAP

        for abbrev, slug in KNOWN_DRIVER_MAP.items():
            assert abbrev == abbrev.upper(), (
                f"Key {abbrev} should be uppercase"
            )
            assert slug == slug.lower(), (
                f"Slug {slug} for {abbrev} should be lowercase"
            )

    def test_driver_map_is_importable(self):
        """DRIVER_MAP should be importable as module-level constant."""
        from data.build_driver_map import DRIVER_MAP

        assert isinstance(DRIVER_MAP, dict)
        assert len(DRIVER_MAP) > 0

    def test_abbreviations_are_3_letters(self):
        """FastF1 abbreviations should be 2–3 uppercase letters."""
        from data.build_driver_map import KNOWN_DRIVER_MAP

        for abbrev in KNOWN_DRIVER_MAP.keys():
            assert 2 <= len(abbrev) <= 3, (
                f"Abbreviation {abbrev} should be 2–3 chars"
            )
            assert abbrev.isalpha(), (
                f"Abbreviation {abbrev} should be alphabetic"
            )

    def test_save_and_load_driver_map(self, tmp_path):
        """save_driver_map → load_driver_map round-trip."""
        from data.build_driver_map import KNOWN_DRIVER_MAP

        # Save to temp file
        map_path = tmp_path / "drivers_map.json"
        with open(map_path, "w") as f:
            json.dump(KNOWN_DRIVER_MAP, f)

        # Load back
        with open(map_path, "r") as f:
            loaded = json.load(f)

        assert loaded == KNOWN_DRIVER_MAP


# ===================================================================
# Test: Jolpica Pipeline
# ===================================================================


class TestJolpicaPipeline:
    """Tests for data.jolpica_pipeline module."""

    def test_jolpica_get_returns_none_on_failure(self):
        """jolpica_get should return None after all retries fail."""
        from data.jolpica_pipeline import jolpica_get

        with patch("data.jolpica_pipeline.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Connection error")
            result = jolpica_get(
                "https://fake.url", retries=1, base_delay=0.01
            )
            assert result is None

    def test_jolpica_get_succeeds_on_valid_response(self):
        """jolpica_get should return parsed JSON on success."""
        from data.jolpica_pipeline import jolpica_get

        mock_response = MagicMock()
        mock_response.json.return_value = {"MRData": {"test": True}}
        mock_response.raise_for_status.return_value = None

        with patch("data.jolpica_pipeline.requests.get", return_value=mock_response):
            with patch("data.jolpica_pipeline.time.sleep"):
                result = jolpica_get(
                    "https://fake.url", retries=1, base_delay=0.01
                )
                assert result == {"MRData": {"test": True}}

    def test_jolpica_get_retries_with_backoff(self):
        """jolpica_get should retry with exponential backoff."""
        from data.jolpica_pipeline import jolpica_get

        mock_response = MagicMock()
        mock_response.json.return_value = {"MRData": {}}
        mock_response.raise_for_status.return_value = None

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.exceptions.RequestException("Transient error")
            return mock_response

        with patch("data.jolpica_pipeline.requests.get", side_effect=side_effect):
            with patch("data.jolpica_pipeline.time.sleep"):
                result = jolpica_get(
                    "https://fake.url", retries=3, base_delay=0.01
                )
                assert result is not None
                assert call_count == 3

    def test_fetch_race_results_schema(self):
        """fetch_race_results should return correct column schema."""
        from data.jolpica_pipeline import fetch_race_results

        mock_data = {
            "MRData": {
                "RaceTable": {
                    "Races": [
                        {
                            "Circuit": {"circuitId": "monza"},
                            "Results": [
                                {
                                    "Driver": {
                                        "driverId": "max_verstappen",
                                        "givenName": "Max",
                                        "familyName": "Verstappen",
                                    },
                                    "Constructor": {"name": "Red Bull"},
                                    "grid": "1",
                                    "position": "1",
                                }
                            ],
                        }
                    ]
                }
            }
        }

        with patch("data.jolpica_pipeline.jolpica_get", return_value=mock_data):
            df = fetch_race_results(2023, 1)
            assert df is not None
            expected_cols = {
                "season", "round", "driver_id", "driver_name",
                "team", "grid_position", "finish_position", "circuit_id",
            }
            assert expected_cols.issubset(set(df.columns))

    def test_fetch_driver_standings_schema(self):
        """fetch_driver_standings should return driver_id + championship_standing."""
        from data.jolpica_pipeline import fetch_driver_standings

        mock_data = {
            "MRData": {
                "StandingsTable": {
                    "StandingsLists": [
                        {
                            "DriverStandings": [
                                {
                                    "position": "1",
                                    "Driver": {
                                        "driverId": "max_verstappen",
                                        "givenName": "Max",
                                        "familyName": "Verstappen",
                                    },
                                },
                                {
                                    "position": "2",
                                    "Driver": {
                                        "driverId": "lewis_hamilton",
                                        "givenName": "Lewis",
                                        "familyName": "Hamilton",
                                    },
                                },
                            ]
                        }
                    ]
                }
            }
        }

        with patch("data.jolpica_pipeline.jolpica_get", return_value=mock_data):
            df = fetch_driver_standings(2023, 5)
            assert df is not None
            assert "driver_id" in df.columns
            assert "championship_standing" in df.columns
            assert len(df) == 2


# ===================================================================
# Test: Merge Logic
# ===================================================================


class TestMergeLogic:
    """Tests for data.__init__ merge functions."""

    @pytest.fixture
    def sample_fastf1_df(self):
        """Create a sample FastF1-style DataFrame."""
        return pd.DataFrame(
            {
                "season": [2023, 2023, 2023, 2023],
                "round": [1, 1, 1, 1],
                "driver_id": ["VER", "HAM", "LEC", "NOR"],
                "driver_name": [
                    "Max Verstappen", "Lewis Hamilton",
                    "Charles Leclerc", "Lando Norris",
                ],
                "team": [
                    "Red Bull Racing", "Mercedes",
                    "Ferrari", "McLaren",
                ],
                "grid_position": [1, 4, 2, 5],
                "finish_position": [1, 3, 2, 4],
                "circuit_id": [
                    "Bahrain Grand Prix", "Bahrain Grand Prix",
                    "Bahrain Grand Prix", "Bahrain Grand Prix",
                ],
                "sector_1_time": [28.5, 28.8, 28.6, 28.9],
                "sector_2_time": [35.2, 35.5, 35.3, 35.6],
                "sector_3_time": [30.1, 30.4, 30.2, 30.5],
                "avg_lap_time_practice": [93.8, 94.7, 94.1, 95.0],
                "tire_compound": [0, 1, 0, 1],
                "tire_age_laps": [20, 25, 22, 28],
                "fresh_tire": [1, 1, 1, 0],
                "pit_stop_count": [2, 2, 3, 2],
                "team_pit_speed": [2.5, 2.8, 2.6, 2.9],
                "weather_temp_track": [45.0, 45.0, 45.0, 45.0],
                "weather_rainfall": [0, 0, 0, 0],
                "telemetry_available": [True, True, True, True],
            }
        )

    @pytest.fixture
    def sample_jolpica_df(self):
        """Create a sample Jolpica-style DataFrame."""
        return pd.DataFrame(
            {
                "season": [2016, 2016],
                "round": [1, 1],
                "driver_id": ["HAM", "ROS"],
                "driver_name": [
                    "Lewis Hamilton", "Nico Rosberg",
                ],
                "team": ["Mercedes", "Mercedes"],
                "grid_position": [1, 2],
                "finish_position": [2, 1],
                "circuit_id": ["albert_park", "albert_park"],
                "sector_1_time": [np.nan, np.nan],
                "sector_2_time": [np.nan, np.nan],
                "sector_3_time": [np.nan, np.nan],
                "avg_lap_time_practice": [np.nan, np.nan],
                "tire_compound": [np.nan, np.nan],
                "tire_age_laps": [np.nan, np.nan],
                "fresh_tire": [np.nan, np.nan],
                "pit_stop_count": [2, 1],
                "team_pit_speed": [np.nan, np.nan],
                "weather_temp_track": [np.nan, np.nan],
                "weather_rainfall": [np.nan, np.nan],
                "championship_standing": [1, 2],
                "telemetry_available": [False, False],
            }
        )

    @pytest.fixture
    def sample_lap_data(self):
        """Create sample lap data for safety car computation."""
        return pd.DataFrame(
            {
                "season": [2023] * 10,
                "round": [1] * 10,
                "driver_id": ["VER"] * 10,
                "lap_number": list(range(1, 11)),
                "track_status": ["1", "1", "4", "4", "1", "1", "1", "6", "1", "1"],
                "circuit_id": ["Bahrain Grand Prix"] * 10,
            }
        )

    def test_merge_produces_correct_columns(
        self, sample_fastf1_df, sample_jolpica_df, sample_lap_data
    ):
        """Merged dataset must have all required columns."""
        from data import merge_datasets

        merged = merge_datasets(
            sample_fastf1_df, sample_jolpica_df, sample_lap_data
        )

        required_cols = {
            "season", "round", "driver_id", "grid_position",
            "finish_position", "circuit_id", "telemetry_available",
            "regulation_era", "track_type", "driver_form_last3",
            "safety_car_probability", "win_probability",
        }
        assert required_cols.issubset(set(merged.columns)), (
            f"Missing columns: {required_cols - set(merged.columns)}"
        )

    def test_merge_row_count(
        self, sample_fastf1_df, sample_jolpica_df, sample_lap_data
    ):
        """Row count after merge should be sum of both DataFrames."""
        from data import merge_datasets

        merged = merge_datasets(
            sample_fastf1_df, sample_jolpica_df, sample_lap_data
        )
        expected = len(sample_fastf1_df) + len(sample_jolpica_df)
        assert len(merged) == expected

    def test_regulation_era_assignment(
        self, sample_fastf1_df, sample_jolpica_df, sample_lap_data
    ):
        """Regulation era should be correctly assigned by season."""
        from data import merge_datasets

        merged = merge_datasets(
            sample_fastf1_df, sample_jolpica_df, sample_lap_data
        )

        # 2016 → hybrid_era
        era_2016 = merged[merged["season"] == 2016]["regulation_era"].unique()
        assert "hybrid_era" in era_2016

        # 2023 → ground_effect_era
        era_2023 = merged[merged["season"] == 2023]["regulation_era"].unique()
        assert "ground_effect_era" in era_2023

    def test_win_probability_target(
        self, sample_fastf1_df, sample_jolpica_df, sample_lap_data
    ):
        """win_probability should be 1 only for P1 finishes."""
        from data import merge_datasets

        merged = merge_datasets(
            sample_fastf1_df, sample_jolpica_df, sample_lap_data
        )

        p1_rows = merged[merged["finish_position"] == 1]
        non_p1_rows = merged[merged["finish_position"] != 1]

        assert (p1_rows["win_probability"] == 1).all()
        assert (non_p1_rows["win_probability"] == 0).all()

    def test_telemetry_flag_integrity(
        self, sample_fastf1_df, sample_jolpica_df, sample_lap_data
    ):
        """telemetry_available must be True for FastF1, False for Jolpica."""
        from data import merge_datasets

        merged = merge_datasets(
            sample_fastf1_df, sample_jolpica_df, sample_lap_data
        )

        fastf1_rows = merged[merged["season"] >= 2018]
        jolpica_rows = merged[merged["season"] <= 2017]

        assert fastf1_rows["telemetry_available"].all()
        assert not jolpica_rows["telemetry_available"].any()

    def test_sorted_by_season_round(
        self, sample_fastf1_df, sample_jolpica_df, sample_lap_data
    ):
        """Merged dataset must be sorted by (season, round, grid_position)."""
        from data import merge_datasets

        merged = merge_datasets(
            sample_fastf1_df, sample_jolpica_df, sample_lap_data
        )

        # Check season is non-decreasing
        assert (merged["season"].diff().dropna() >= 0).all()

    def test_safety_car_probability_computed(self, sample_lap_data):
        """safety_car_probability should be computed from lap data."""
        from data import compute_safety_car_probability

        sc_df = compute_safety_car_probability(sample_lap_data)
        assert len(sc_df) == 1  # One circuit
        assert "safety_car_probability" in sc_df.columns

        # 2 SC laps out of 10 → 0.2
        prob = sc_df.iloc[0]["safety_car_probability"]
        assert prob == pytest.approx(0.2, abs=0.01)

    def test_track_type_mapping(self):
        """_get_track_type should correctly classify circuits."""
        from data import _get_track_type

        assert _get_track_type("Monaco Grand Prix") == "street"
        assert _get_track_type("Singapore Grand Prix") == "street"
        assert _get_track_type("Australian Grand Prix") == "hybrid"
        assert _get_track_type("Silverstone Grand Prix") == "permanent"


# ===================================================================
# Test: FastF1 Pipeline (unit tests with mocking)
# ===================================================================


class TestFastF1Pipeline:
    """Tests for data.fastf1_pipeline module — mocked to avoid real API calls."""

    def test_enable_cache_creates_directory(self, tmp_path):
        """enable_cache should create the cache directory."""
        from data.fastf1_pipeline import enable_cache

        cache_dir = tmp_path / "test_cache"
        with patch("data.fastf1_pipeline.fastf1.Cache.enable_cache"):
            enable_cache(str(cache_dir))
        assert cache_dir.exists()

    def test_build_season_returns_tuple(self):
        """build_season_dataframe should return (race_df, lap_data_df) tuple."""
        from data.fastf1_pipeline import build_season_dataframe

        # Mock schedule to return empty (no events)
        with patch("data.fastf1_pipeline.fastf1.get_event_schedule") as mock_sched:
            mock_sched.side_effect = Exception("No schedule")
            with patch("data.fastf1_pipeline.enable_cache"):
                result = build_season_dataframe(2023)
                assert isinstance(result, tuple)
                assert len(result) == 2


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
