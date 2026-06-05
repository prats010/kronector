"""
Tests for Week 2 feature engineering.

Run: python -m pytest tests/test_feature_engineering.py -v
"""

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import (
    add_driver_form,
    add_era_normalized_sector_times,
    create_time_series_splits,
    load_encoders,
    prepare_model_data,
    save_encoders,
    validate_input_schema,
)


@pytest.fixture
def sample_race_df():
    return pd.DataFrame(
        {
            "season": [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            "round": [1, 2, 1, 2, 1, 2, 1, 2],
            "driver_id": ["VER", "HAM", "VER", "HAM", "VER", "HAM", "VER", "HAM"],
            "driver_name": [
                "Max Verstappen",
                "Lewis Hamilton",
                "Max Verstappen",
                "Lewis Hamilton",
                "Max Verstappen",
                "Lewis Hamilton",
                "Max Verstappen",
                "Lewis Hamilton",
            ],
            "team": [
                "Red Bull Racing",
                "Mercedes",
                "Red Bull Racing",
                "Mercedes",
                "Red Bull Racing",
                "Mercedes",
                "Red Bull Racing",
                None,
            ],
            "grid_position": [1, 2, 1, 3, 1, 4, 2, 1],
            "finish_position": [1, 2, 1, 3, 2, 1, 1, 2],
            "circuit_id": [
                "Monaco Grand Prix",
                "Silverstone Grand Prix",
                "Bahrain Grand Prix",
                "Australian Grand Prix",
                "Singapore Grand Prix",
                "Italian Grand Prix",
                "Canadian Grand Prix",
                "Las Vegas Grand Prix",
            ],
            "regulation_era": [
                "hybrid_era",
                "hybrid_era",
                "ground_effect_era",
                "ground_effect_era",
                "ground_effect_era",
                "ground_effect_era",
                "ground_effect_era",
                "ground_effect_era",
            ],
            "track_type": [
                "street",
                "permanent",
                "permanent",
                "hybrid",
                "street",
                "permanent",
                "hybrid",
                "street",
            ],
            "sector_1_time": [28.0, np.nan, 29.0, 30.0, 28.5, 29.2, 28.8, 29.1],
            "sector_2_time": [35.0, 35.5, 36.0, np.nan, 35.8, 36.1, 35.6, 35.9],
            "sector_3_time": [22.0, 22.2, np.nan, 22.8, 22.5, 22.7, 22.4, 22.6],
            "avg_lap_time_practice": [93.0, np.nan, 94.0, 95.0, 93.5, 94.2, 93.8, 94.1],
            "tire_compound": [0, 1, 0, 1, np.nan, 1, 2, 0],
            "tire_age_laps": [14, 12, 18, 16, 20, np.nan, 15, 17],
            "fresh_tire": [1, 1, 1, 0, 1, 1, np.nan, 0],
            "pit_stop_count": [2, 1, 2, 2, 3, 2, 1, np.nan],
            "team_pit_speed": [np.nan] * 8,
            "weather_temp_track": [30.0, 28.0, 31.0, 27.0, 29.0, 32.0, 26.0, 25.0],
            "weather_rainfall": [0, 0, 0, 1, 0, 0, 1, 0],
            "championship_standing": [1, 2, 1, 3, 1, 2, 1, 2],
            "driver_form_last3": [np.nan, np.nan, 1.0, 2.0, 1.0, 2.5, 1.3, 2.3],
            "safety_car_probability": [0.2, 0.0, 0.1, 0.3, 0.4, 0.1, 0.2, 0.5],
            "telemetry_available": [True] * 8,
            "win_probability": [1, 0, 1, 0, 0, 1, 1, 0],
        }
    )


def test_validate_input_schema_rejects_missing_required_column(sample_race_df):
    broken = sample_race_df.drop(columns=["driver_id"])

    with pytest.raises(ValueError, match="driver_id"):
        validate_input_schema(broken)


def test_prepare_model_data_imputes_missing_values(sample_race_df):
    bundle, encoders = prepare_model_data(sample_race_df)

    assert not bundle.X.isna().any().any()
    assert len(bundle.X) == len(sample_race_df)
    assert len(bundle.y) == len(sample_race_df)
    assert set(encoders) == {"team", "track_type", "regulation_era"}


def test_prepare_model_data_excludes_leakage_columns(sample_race_df):
    bundle, _ = prepare_model_data(sample_race_df)

    assert "finish_position" not in bundle.feature_columns
    assert "driver_id" not in bundle.feature_columns
    assert "win_probability" not in bundle.feature_columns
    assert "round" not in bundle.feature_columns
    assert "round" in bundle.metadata.columns


def test_prepare_model_data_label_encodes_categoricals(sample_race_df):
    bundle, encoders = prepare_model_data(sample_race_df)

    assert "team" in bundle.feature_columns
    assert "track_type" in bundle.feature_columns
    assert "regulation_era" in bundle.feature_columns
    assert "unknown" in encoders["team"].classes_
    assert pd.api.types.is_integer_dtype(bundle.X["team"])


def test_era_normalized_sector_times_center_within_era(sample_race_df):
    normalized = add_era_normalized_sector_times(sample_race_df)

    for era, group in normalized.groupby("regulation_era"):
        mean_value = group["sector_1_time_era_norm"].mean()
        assert mean_value == pytest.approx(0.0, abs=1e-12), era


def test_time_series_splits_are_chronological(sample_race_df):
    bundle, _ = prepare_model_data(sample_race_df)
    splits = list(create_time_series_splits(bundle.X, n_splits=3))

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()


def test_prepare_model_data_can_accept_fastf1_only_shape(sample_race_df):
    fastf1_only = sample_race_df.drop(
        columns=[
            "regulation_era",
            "track_type",
            "championship_standing",
            "driver_form_last3",
            "safety_car_probability",
            "win_probability",
        ]
    )

    bundle, _ = prepare_model_data(fastf1_only)

    assert "win_probability" not in bundle.feature_columns
    assert set(bundle.y.unique()) <= {0, 1}
    assert "safety_car_probability" in bundle.feature_columns


def test_prepare_model_data_reuses_fitted_encoders_for_inference(sample_race_df):
    train_df = sample_race_df.iloc[:6].copy()
    inference_df = sample_race_df.iloc[6:].copy()
    inference_df.loc[:, "team"] = "New Team"

    _, encoders = prepare_model_data(train_df)
    inference_bundle, reused_encoders = prepare_model_data(inference_df, encoders)

    unknown_code = encoders["team"].transform(["unknown"])[0]
    assert reused_encoders is encoders
    assert (inference_bundle.X["team"] == unknown_code).all()


def test_add_driver_form_uses_only_prior_races():
    df = pd.DataFrame(
        {
            "driver_id": ["VER", "VER", "VER", "VER"],
            "season": [2023, 2023, 2023, 2023],
            "round": [1, 2, 3, 4],
            "finish_position": [1, 3, 5, 7],
        }
    )

    result = add_driver_form(df)

    assert pd.isna(result.loc[0, "driver_form_last3"])
    assert result.loc[1, "driver_form_last3"] == pytest.approx(1.0)
    assert result.loc[2, "driver_form_last3"] == pytest.approx(2.0)
    assert result.loc[3, "driver_form_last3"] == pytest.approx(3.0)


def test_prepare_model_data_computes_driver_form_only_when_missing(sample_race_df):
    missing_form = sample_race_df.copy()
    missing_form["driver_form_last3"] = np.nan

    bundle, _ = prepare_model_data(missing_form)

    assert not bundle.X["driver_form_last3"].isna().any()


def test_championship_standing_uses_season_max_fill(sample_race_df):
    df = sample_race_df.copy()
    df.loc[(df["season"] == 2023) & (df["driver_id"] == "HAM"), "championship_standing"] = np.nan

    bundle, _ = prepare_model_data(df)
    row = bundle.metadata[
        (bundle.metadata["season"] == 2023) & (bundle.metadata["driver_id"] == "HAM")
    ].index[0]

    assert bundle.X.loc[row, "championship_standing"] == 1


def test_encoder_persistence_round_trip(sample_race_df, tmp_path):
    _, encoders = prepare_model_data(sample_race_df)
    path = tmp_path / "label_encoders.pkl"

    save_encoders(encoders, str(path))
    loaded = load_encoders(str(path))

    assert set(loaded) == set(encoders)
    assert list(loaded["team"].classes_) == list(encoders["team"].classes_)
