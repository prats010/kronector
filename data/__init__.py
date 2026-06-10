"""
KRONECTOR — Data Package
Merge logic for combining FastF1 (2018–2024) and Jolpica (2014–2017) datasets.

Merge key: (season, round, driver_id)
Master driver_id format: FastF1 3-letter abbreviation (e.g., VER, HAM)

This module:
  1. Aligns column schemas between both sources
  2. Merges championship_standing from Jolpica onto all rows (sole source)
  3. Computes safety_car_probability from lap data per circuit
  4. Computes driver_form_last3 (rolling avg finish, last 3 races)
  5. Adds regulation_era and track_type features
  6. Adds win_probability target variable
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit → track_type mapping
# ---------------------------------------------------------------------------

TRACK_TYPE_MAP = {
    # Street circuits
    "monaco": "street",
    "Monaco Grand Prix": "street",
    "baku": "street",
    "Azerbaijan Grand Prix": "street",
    "marina_bay": "street",
    "Singapore Grand Prix": "street",
    "vegas": "street",
    "Las Vegas Grand Prix": "street",
    "jeddah": "street",
    "Saudi Arabian Grand Prix": "street",
    # Hybrid circuits (semi-permanent / street-like sections)
    "albert_park": "hybrid",
    "Australian Grand Prix": "hybrid",
    "villeneuve": "hybrid",
    "Canadian Grand Prix": "hybrid",
    "sochi": "hybrid",
    "Russian Grand Prix": "hybrid",
    "miami": "hybrid",
    "Miami Grand Prix": "hybrid",
    # Everything else → permanent
}


def _get_track_type(circuit_id: str) -> str:
    """Determine track type from circuit identifier."""
    circuit_lower = str(circuit_id).lower()
    for key, track_type in TRACK_TYPE_MAP.items():
        if key.lower() in circuit_lower:
            return track_type
    return "permanent"


def compute_pole_conversion_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical pole-to-win conversion rate per circuit.

    For each circuit, calculates what % of races the pole-sitter went on to win.
    This directly encodes "qualifying matters at this track" — e.g.:
      - Monaco ~75-80% (almost impossible to overtake)
      - Monza ~40-45% (long straights, slipstream, DRS)

    Uses only COMPLETED races (requires finish_position data).

    Args:
        df: Historical race DataFrame with circuit_id, grid_position, finish_position

    Returns:
        DataFrame with columns (circuit_id, pole_conversion_rate)
    """
    if df.empty or "finish_position" not in df.columns:
        return pd.DataFrame(columns=["circuit_id", "pole_conversion_rate"])

    # Only use rows where we have actual race results
    completed = df.dropna(subset=["finish_position"]).copy()

    # Get pole-sitters (grid_position == 1) for each race
    poles = completed[completed["grid_position"] == 1].copy()

    if poles.empty:
        return pd.DataFrame(columns=["circuit_id", "pole_conversion_rate"])

    poles["pole_won"] = (poles["finish_position"] == 1).astype(int)

    global_mean = poles["pole_won"].mean()
    C = 3.0  # Confidence weight (pseudo-observations)

    # Calculate wins and total poles per circuit
    stats = poles.groupby("circuit_id").agg(
        wins=("pole_won", "sum"),
        total=("pole_won", "count")
    ).reset_index()

    # Apply Bayesian smoothing: pulls low-N circuits towards the global average
    stats["pole_conversion_rate"] = (stats["wins"] + C * global_mean) / (stats["total"] + C)
    
    conversion = stats[["circuit_id", "pole_conversion_rate"]]

    logger.info(
        f"Computed pole_conversion_rate for {len(conversion)} circuits. "
        f"Range: {conversion['pole_conversion_rate'].min():.1%} – "
        f"{conversion['pole_conversion_rate'].max():.1%}"
    )

    return conversion

    return conversion


# ---------------------------------------------------------------------------
# Career Race Starts computation
# ---------------------------------------------------------------------------

def compute_career_race_starts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative career race starts for each driver before each race.
    This provides an 'experience' feature so the model knows if driver form
    is based on a veteran's long track record or a rookie's small sample size.
    """
    if df.empty or "driver_id" not in df.columns:
        return pd.DataFrame(columns=["season", "round", "driver_id", "career_race_starts"])
        
    # Sort chronologically
    df_sorted = df.sort_values(by=["driver_id", "season", "round"]).copy()
    
    # cumcount() gives 0 for the 1st race, 1 for the 2nd, etc.
    # This exactly represents 'starts prior to this race'
    df_sorted["career_race_starts"] = df_sorted.groupby("driver_id").cumcount()
    
    return df_sorted[["season", "round", "driver_id", "career_race_starts"]]


# ---------------------------------------------------------------------------
# Safety car probability computation (Correction 2)
# ---------------------------------------------------------------------------

def compute_safety_car_probability(
    lap_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute safety car probability per circuit from historical lap data.

    Correction 2: Uses fetch_lap_data output.
    - Group by circuit_id across all historical seasons
    - sc_laps = count of laps where track_status == '4'
    - total_laps = total laps at that circuit
    - safety_car_probability = sc_laps / total_laps

    Args:
        lap_data: DataFrame from fastf1_pipeline.fetch_lap_data with columns
                  (season, round, driver_id, lap_number, track_status, circuit_id)

    Returns:
        DataFrame with columns (circuit_id, safety_car_probability)
    """
    if lap_data.empty:
        logger.warning("No lap data available for safety car computation")
        return pd.DataFrame(columns=["circuit_id", "safety_car_probability"])

    # Deduplicate: one entry per (circuit_id, season, round, lap_number)
    # Multiple drivers may have different track_status on same lap;
    # any driver seeing SC means SC was deployed
    lap_circuit = lap_data.copy()

    # Group by circuit
    circuit_stats = []
    for circuit_id, group in lap_circuit.groupby("circuit_id"):
        # Unique laps per race (not per driver)
        race_laps = group.drop_duplicates(
            subset=["season", "round", "lap_number"]
        )
        total_laps = len(race_laps)
        sc_laps = len(
            race_laps[race_laps["track_status"].astype(str) == "4"]
        )

        sc_prob = sc_laps / total_laps if total_laps > 0 else 0.0

        circuit_stats.append(
            {
                "circuit_id": circuit_id,
                "safety_car_probability": round(sc_prob, 4),
            }
        )

    result = pd.DataFrame(circuit_stats)
    logger.info(
        f"Computed safety_car_probability for {len(result)} circuits"
    )
    return result


# ---------------------------------------------------------------------------
# Driver form computation (rolling avg finish, last 3 races)
# ---------------------------------------------------------------------------

def compute_driver_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute driver_form_last3: rolling average finish position over last 3 races.

    The DataFrame MUST be sorted by (season, round) before calling this.
    Computes per driver using shift to avoid data leakage (only past races).

    Args:
        df: Merged DataFrame sorted by (season, round, grid_position)

    Returns:
        Same DataFrame with driver_form_last3 column added
    """
    df = df.sort_values(["season", "round", "driver_id"]).copy()

    # Compute rolling mean of finish_position per driver (last 3 races)
    df["driver_form_last3"] = (
        df.groupby("driver_id")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )

    return df


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------

def merge_datasets(
    fastf1_df: pd.DataFrame,
    jolpica_df: pd.DataFrame,
    lap_data: pd.DataFrame,
    standings_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge FastF1 and Jolpica datasets into a unified dataset.

    Steps:
      1. Align column schemas (union of all columns)
      2. pd.concat([jolpica_df, fastf1_df])
      3. Sort by (season, round, grid_position)
      4. Add regulation_era: 2014–2021 → hybrid_era, 2022–2024 → ground_effect_era
      5. Add track_type from circuit mapping
      6. Join championship_standing from Jolpica (sole source — Correction 1)
      7. Compute driver_form_last3 (rolling avg finish, last 3 races)
      8. Compute safety_car_probability from lap data (Correction 2)
      9. Add win_probability target (1 if finish_position == 1, else 0)
      10. Validate telemetry_available flag integrity

    Args:
        fastf1_df: DataFrame from fastf1_pipeline (2018–2024)
        jolpica_df: DataFrame from jolpica_pipeline (2014–2017)
        lap_data: DataFrame from fastf1_pipeline.fetch_lap_data
        standings_df: Optional championship standings from Jolpica for all years

    Returns:
        Unified DataFrame ready for feature engineering
    """
    logger.info("Starting dataset merge...")

    fastf1_df = fastf1_df.copy()
    jolpica_df = jolpica_df.copy()

    # -------------------------------------------------------------------
    # Step 1–2: Align schemas and concatenate
    # -------------------------------------------------------------------
    # Ensure both DataFrames have the same columns
    all_columns = set(fastf1_df.columns) | set(jolpica_df.columns)

    for col in all_columns:
        if col not in fastf1_df.columns:
            fastf1_df[col] = np.nan
        if col not in jolpica_df.columns:
            jolpica_df[col] = np.nan

    # Reorder columns to match
    col_order = sorted(all_columns)
    fastf1_df = fastf1_df[col_order]
    jolpica_df = jolpica_df[col_order]

    merged = pd.concat([jolpica_df, fastf1_df], ignore_index=True)
    logger.info(
        f"Concatenated: {len(jolpica_df)} Jolpica + {len(fastf1_df)} FastF1 "
        f"= {len(merged)} total rows"
    )

    # -------------------------------------------------------------------
    # Step 3: Sort chronologically
    # -------------------------------------------------------------------
    merged = merged.sort_values(
        ["season", "round", "grid_position"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------
    # Step 4: Add regulation_era
    # -------------------------------------------------------------------
    merged["regulation_era"] = merged["season"].apply(
        lambda s: "agile_era" if s >= 2026 else ("ground_effect_era" if s >= 2022 else "hybrid_era")
    )

    # -------------------------------------------------------------------
    # Step 5: Add track_type
    # -------------------------------------------------------------------
    merged["track_type"] = merged["circuit_id"].apply(_get_track_type)

    # -------------------------------------------------------------------
    # Step 6: Join championship_standing (Correction 1 — sole source: Jolpica)
    # -------------------------------------------------------------------
    if standings_df is not None and not standings_df.empty:
        # Drop any existing championship_standing before merge
        if "championship_standing" in merged.columns:
            # Keep Jolpica-sourced standings from backfill rows
            fastf1_mask = merged["telemetry_available"] == True  # noqa: E712
            merged.loc[fastf1_mask, "championship_standing"] = np.nan
        else:
            merged["championship_standing"] = np.nan

        # Merge standings for FastF1 rows
        standings_cols = ["season", "round", "driver_id", "championship_standing"]
        standings_clean = standings_df[standings_cols].drop_duplicates(
            subset=["season", "round", "driver_id"]
        )

        # Only merge onto rows that don't already have standings
        needs_standings = merged["championship_standing"].isna()
        
        logger.debug(f"Driver IDs in merged: {set(merged.get('driver_id', []))}")
        logger.debug(f"Driver IDs in standings: {set(standings_clean.get('driver_id', []))}")
        
        if needs_standings.any():
            merged = merged.merge(
                standings_clean,
                on=["season", "round", "driver_id"],
                how="left",
                suffixes=("", "_jolpica"),
            )
            # Fill NaN championship_standing with Jolpica values
            if "championship_standing_jolpica" in merged.columns:
                merged["championship_standing"] = merged[
                    "championship_standing"
                ].fillna(merged["championship_standing_jolpica"])
                merged = merged.drop(columns=["championship_standing_jolpica"])

        logger.info("Joined championship_standing from Jolpica")
    else:
        logger.warning(
            "No standings_df provided — championship_standing may be incomplete"
        )

    # -------------------------------------------------------------------
    # Step 7: Compute driver_form_last3
    # -------------------------------------------------------------------
    merged = compute_driver_form(merged)
    logger.info("Computed driver_form_last3")

    # -------------------------------------------------------------------
    # Step 8: Compute safety_car_probability (Correction 2)
    # -------------------------------------------------------------------
    if not lap_data.empty:
        sc_prob = compute_safety_car_probability(lap_data)
        if not sc_prob.empty:
            if "safety_car_probability" in merged.columns:
                merged = merged.drop(columns=["safety_car_probability"])
            merged = merged.merge(sc_prob, on="circuit_id", how="left")
            # Fill circuits without lap data (Jolpica years) with 0
            merged["safety_car_probability"] = merged.get("safety_car_probability", pd.Series(np.nan, index=merged.index)).fillna(0.0)
            logger.info("Joined safety_car_probability from lap data")
    else:
        merged["safety_car_probability"] = 0.0
        logger.warning(
            "No lap data — safety_car_probability set to 0 for all rows"
        )

    # -------------------------------------------------------------------
    # Step 8.3: Compute career_race_starts per driver
    # -------------------------------------------------------------------
    starts_df = compute_career_race_starts(merged)
    if not starts_df.empty:
        if "career_race_starts" in merged.columns:
            merged = merged.drop(columns=["career_race_starts"])
        merged = merged.merge(starts_df, on=["season", "round", "driver_id"], how="left")
        merged["career_race_starts"] = merged["career_race_starts"].fillna(0)
        logger.info("Joined career_race_starts from cumulative history")
    else:
        merged["career_race_starts"] = 0
        logger.warning("Could not compute career_race_starts — defaulting to 0")

    # -------------------------------------------------------------------
    # Step 8.5: Compute pole_conversion_rate per circuit
    # -------------------------------------------------------------------
    pcr = compute_pole_conversion_rate(merged)
    if not pcr.empty:
        if "pole_conversion_rate" in merged.columns:
            merged = merged.drop(columns=["pole_conversion_rate"])
        merged = merged.merge(pcr, on="circuit_id", how="left")
        # Circuits with no data default to 50% (neutral)
        merged["pole_conversion_rate"] = merged["pole_conversion_rate"].fillna(0.5)
        logger.info("Joined pole_conversion_rate from historical data")
    else:
        merged["pole_conversion_rate"] = 0.5
        logger.warning("Could not compute pole_conversion_rate — defaulting to 0.5")

    # -------------------------------------------------------------------
    # Step 9: Add target variable
    # -------------------------------------------------------------------
    merged["win_probability"] = (
        merged["finish_position"] == 1
    ).astype(int)

    # -------------------------------------------------------------------
    # Step 10: Validate telemetry_available flag
    # -------------------------------------------------------------------
    # Fix PyArrow Parquet conversion error by explicitly casting to boolean
    merged["telemetry_available"] = merged["telemetry_available"].fillna(False).astype(bool)
    
    fastf1_rows = merged[merged["telemetry_available"] == True]  # noqa: E712
    jolpica_rows = merged[merged["telemetry_available"] == False]  # noqa: E712

    # FastF1 rows should have season >= 2018
    bad_fastf1 = fastf1_rows[fastf1_rows["season"] < 2018]
    if not bad_fastf1.empty:
        logger.error(
            f"INTEGRITY ERROR: {len(bad_fastf1)} rows with "
            f"telemetry_available=True but season < 2018"
        )

    # Jolpica rows should have season <= 2017
    bad_jolpica = jolpica_rows[jolpica_rows["season"] > 2017]
    if not bad_jolpica.empty:
        logger.warning(
            f"Note: {len(bad_jolpica)} Jolpica rows with season > 2017 "
            f"(standings-only merge — expected)"
        )

    logger.info(
        f"\nMerge complete: {len(merged)} total rows\n"
        f"  FastF1 (telemetry=True):  {len(fastf1_rows)}\n"
        f"  Jolpica (telemetry=False): {len(jolpica_rows)}\n"
        f"  Seasons: {merged['season'].min()}–{merged['season'].max()}\n"
        f"  Races: {merged.groupby(['season', 'round']).ngroups}\n"
        f"  Win rate: {merged['win_probability'].mean():.3f}"
    )

    return merged


# ---------------------------------------------------------------------------
# Convenience: full pipeline runner
# ---------------------------------------------------------------------------

def run_full_pipeline(
    fastf1_start: int = 2018,
    fastf1_end: int = 2026,
    jolpica_start: int = 2014,
    jolpica_end: int = 2017,
) -> pd.DataFrame:
    """
    Run the complete data pipeline: FastF1 + Jolpica + merge.

    This is the main entry point for building the full dataset.

    Returns:
        Unified DataFrame ready for feature engineering.
    """
    from data.fastf1_pipeline import build_full_dataset as build_fastf1
    from data.jolpica_pipeline import (
        build_jolpica_dataset,
        fetch_all_standings,
    )

    # 1. Build FastF1 dataset
    logger.info("Step 1: Building FastF1 dataset...")
    fastf1_df, lap_data = build_fastf1(fastf1_start, fastf1_end)

    # 2. Build Jolpica backfill
    logger.info("Step 2: Building Jolpica dataset...")
    jolpica_df = build_jolpica_dataset(jolpica_start, jolpica_end)

    # 3. Fetch all standings from Jolpica (sole source)
    logger.info("Step 3: Fetching all championship standings...")
    standings_df = fetch_all_standings(fastf1_start, fastf1_end)

    # 4. Merge
    logger.info("Step 4: Merging datasets...")
    merged = merge_datasets(fastf1_df, jolpica_df, lap_data, standings_df)

    return merged
