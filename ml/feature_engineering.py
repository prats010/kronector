"""
KRONECTOR - Feature engineering for model training.

This module turns the merged race dataset into numeric model inputs while
keeping chronological ordering intact for time-series validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

try:
    from data import _get_track_type
except ImportError:  # pragma: no cover - defensive fallback for isolated use
    _get_track_type = None


TARGET_COLUMN = "win_probability"

BASE_REQUIRED_COLUMNS = {
    "season",
    "round",
    "driver_id",
    "team",
    "grid_position",
    "finish_position",
    "circuit_id",
}

SECTOR_COLUMNS = ["sector_1_time", "sector_2_time", "sector_3_time"]

NUMERIC_FEATURES = [
    "season",
    "grid_position",
    "sector_1_time",
    "sector_2_time",
    "sector_3_time",
    "sector_1_time_era_norm",
    "sector_2_time_era_norm",
    "sector_3_time_era_norm",
    "avg_lap_time_practice",
    "tire_compound",
    "tire_age_laps",
    "fresh_tire",
    "pit_stop_count",
    "team_pit_speed",
    "weather_temp_track",
    "weather_rainfall",
    "championship_standing",
    "driver_form_last3",
    "safety_car_probability",
    "telemetry_available",
]

CATEGORICAL_FEATURES = ["team", "track_type", "regulation_era"]
UNKNOWN_CATEGORY = "unknown"

LEAKAGE_COLUMNS = {
    "finish_position",
    "driver_name",
    "driver_id",
    "circuit_id",
    TARGET_COLUMN,
}

EXCLUDED_FEATURE_COLUMNS = LEAKAGE_COLUMNS | {"round"}


@dataclass(frozen=True)
class FeatureBundle:
    """Container returned by prepare_model_data."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    feature_columns: list[str]


def validate_input_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the minimum training schema is missing."""
    missing = BASE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def ensure_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived/default columns expected by feature engineering.

    The preferred input is the merged dataset from data.merge_datasets. This
    helper also accepts the current FastF1-only parquet for smoke training.
    """
    validate_input_schema(df)
    result = df.copy()

    if TARGET_COLUMN not in result.columns:
        result[TARGET_COLUMN] = (result["finish_position"] == 1).astype(int)

    if "regulation_era" not in result.columns:
        result["regulation_era"] = np.where(
            result["season"] >= 2026, "agile_era",
            np.where(result["season"] >= 2022, "ground_effect_era", "hybrid_era")
        )

    if "track_type" not in result.columns:
        if _get_track_type is None:
            result["track_type"] = "permanent"
        else:
            result["track_type"] = result["circuit_id"].apply(_get_track_type)

    defaults = {
        "championship_standing": np.nan,
        "driver_form_last3": np.nan,
        "safety_car_probability": 0.0,
        "telemetry_available": False,
        "avg_lap_time_practice": np.nan,
        "tire_compound": np.nan,
        "tire_age_laps": np.nan,
        "fresh_tire": np.nan,
        "pit_stop_count": np.nan,
        "team_pit_speed": np.nan,
        "weather_temp_track": np.nan,
        "weather_rainfall": np.nan,
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default

    for column in SECTOR_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    return result


def add_era_normalized_sector_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add z-scored sector columns normalized within regulation era.

    Normalizing within era avoids mixing hybrid-era and ground-effect-era lap
    profiles. Zero standard deviation is treated as 1.0 to avoid division by 0.
    """
    result = df.copy()

    for column in SECTOR_COLUMNS:
        norm_column = column.replace("_time", "_time_era_norm")
        grouped = result.groupby("regulation_era")[column]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, 1.0).fillna(1.0)
        result[norm_column] = (result[column] - mean) / std

    return result


def add_driver_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute driver_form_last3 without leaking the current race result.

    The calculation sorts by (driver_id, season, round), then uses
    shift(1).rolling(3).mean() so each row only sees prior races.
    """
    result = df.copy().reset_index(drop=True)
    sorted_df = result.sort_values(["driver_id", "season", "round"]).copy()
    form = (
        sorted_df.groupby("driver_id")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    result.loc[sorted_df.index, "driver_form_last3"] = form
    return result


def _impute_championship_standing(result: pd.DataFrame) -> pd.DataFrame:
    """Fill missing standings with the worst known standing in that season."""
    result["championship_standing"] = pd.to_numeric(
        result["championship_standing"], errors="coerce"
    )
    result["championship_standing"] = result.groupby("season")[
        "championship_standing"
    ].transform(lambda x: x.fillna(x.max()))

    if result["championship_standing"].isna().any():
        global_max = result["championship_standing"].max()
        fill_value = 0.0 if pd.isna(global_max) else global_max
        result["championship_standing"] = result[
            "championship_standing"
        ].fillna(fill_value)

    return result


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric and categorical missing values deterministically."""
    result = df.copy()
    result = _impute_championship_standing(result)

    for column in NUMERIC_FEATURES:
        if column not in result.columns:
            result[column] = np.nan

        if result[column].dtype == bool:
            result[column] = result[column].astype(int)
            continue

        result[column] = pd.to_numeric(result[column], errors="coerce")
        valid_values = result[column].dropna()
        if valid_values.empty:
            median = 0.0
        else:
            median = valid_values.median()
        result[column] = result[column].fillna(median)

    for column in CATEGORICAL_FEATURES:
        if column not in result.columns:
            result[column] = UNKNOWN_CATEGORY
        result[column] = result[column].fillna(UNKNOWN_CATEGORY).astype(str)

    return result


def fit_label_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Fit LabelEncoders for all configured categorical features."""
    encoders = {}
    for column in CATEGORICAL_FEATURES:
        values = df[column].fillna(UNKNOWN_CATEGORY).astype(str)
        values = pd.concat([values, pd.Series([UNKNOWN_CATEGORY])], ignore_index=True)
        encoder = LabelEncoder()
        encoder.fit(values)
        encoders[column] = encoder
    return encoders


def encode_categoricals(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder] | None = None
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode categorical features.

    If encoders are provided, they are reused for inference. Unknown inference
    values are mapped to the explicit "unknown" class fitted during training.
    """
    result = df.copy()
    fitted_encoders = encoders or fit_label_encoders(result)

    for column in CATEGORICAL_FEATURES:
        if column not in fitted_encoders:
            raise ValueError(f"Missing fitted encoder for categorical column: {column}")

        encoder = fitted_encoders[column]
        known_classes = set(encoder.classes_)
        values = result[column].fillna(UNKNOWN_CATEGORY).astype(str)
        values = values.where(values.isin(known_classes), UNKNOWN_CATEGORY)
        result[column] = encoder.transform(values)

    return result, fitted_encoders


def save_encoders(encoders: dict[str, LabelEncoder], path: str) -> None:
    """Persist fitted categorical encoders for model inference."""
    with open(path, "wb") as file:
        pickle.dump(encoders, file)


def load_encoders(path: str) -> dict[str, LabelEncoder]:
    """Load fitted categorical encoders saved by save_encoders."""
    with open(path, "rb") as file:
        return pickle.load(file)


def prepare_model_data(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder] | None = None
) -> tuple[FeatureBundle, dict[str, LabelEncoder]]:
    """
    Build model-ready X/y from a race dataset.

    The returned frame is sorted by (season, round, grid_position), and leakage
    columns such as finish_position are excluded from X.
    """
    prepared = ensure_training_columns(df)
    prepared = prepared.sort_values(["season", "round", "grid_position"]).reset_index(
        drop=True
    )
    if prepared["driver_form_last3"].isna().all():
        prepared = add_driver_form(prepared)

    prepared = add_era_normalized_sector_times(prepared)
    prepared = impute_missing_values(prepared)

    metadata_columns = [
        column
        for column in ["season", "round", "driver_id", "driver_name", "team"]
        if column in prepared.columns
    ]
    metadata = prepared[metadata_columns].copy()

    prepared, fitted_encoders = encode_categoricals(prepared, encoders)

    y = prepared[TARGET_COLUMN].astype(int)
    feature_columns = [
        column
        for column in prepared.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(prepared[column])
    ]
    X = prepared[feature_columns].copy()

    return (
        FeatureBundle(
            X=X,
            y=y,
            metadata=metadata,
            feature_columns=feature_columns,
        ),
        fitted_encoders,
    )


def create_time_series_splits(
    X: pd.DataFrame, n_splits: int = 5
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Return chronological TimeSeriesSplit indices."""
    if len(X) <= n_splits:
        raise ValueError(
            f"Need more rows than n_splits; got {len(X)} rows and {n_splits} splits"
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    return splitter.split(X)
