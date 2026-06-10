"""
KRONECTOR — FastF1 Data Pipeline
Fetches telemetry + session data for F1 seasons 2018–2024.

Data sources:
  - Race session: finishing order, grid positions, lap data, pit stops
  - Qualifying session: sector times (S1, S2, S3)
  - Practice sessions: FP2/FP3 average lap times
  - Weather: track temperature, rainfall

All sector times are RAW seconds — normalization happens in feature_engineering.py.
championship_standing comes from Jolpica pipeline only — NOT fetched here.
"""

import logging
import os
from pathlib import Path
from typing import Optional
import builtins
import requests
import urllib3
builtins.RequestsCookieJar = requests.cookies.RequestsCookieJar
builtins.HTTPAdapter = requests.adapters.HTTPAdapter
builtins.Retry = urllib3.util.Retry

import fastf1
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

def enable_cache(cache_dir: Optional[str] = None) -> None:
    """Configure FastF1 cache directory. Creates dir if not exists."""
    cache_path = cache_dir or os.getenv("FASTF1_CACHE_DIR", "./cache/fastf1")
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_path)
    logger.info(f"FastF1 cache enabled at: {cache_path}")


# ---------------------------------------------------------------------------
# Session loader helper
# ---------------------------------------------------------------------------

def _load_session(
    season: int, round_num: int, session_type: str,
    laps: bool = True, telemetry: bool = False,
    weather: bool = True, messages: bool = False,
) -> Optional[fastf1.core.Session]:
    """
    Load a FastF1 session with error handling.

    Args:
        season: F1 season year (2018–2026)
        round_num: Race round number within season
        session_type: One of 'R' (Race), 'Q' (Qualifying), 'FP2', 'FP3'
        laps: Load lap timing data (default True)
        telemetry: Load car telemetry (default False — not needed for features)
        weather: Load weather data (default True)
        messages: Load race control messages (default False)

    Returns:
        Loaded FastF1 Session object, or None if unavailable.
    """
    try:
        session = fastf1.get_session(season, round_num, session_type)
        session.load(laps=laps, telemetry=telemetry, weather=weather, messages=messages)
        return session
    except Exception as e:
        logger.warning(
            f"Could not load {session_type} session for "
            f"{season} R{round_num}: {e}"
        )
        return None


# ---------------------------------------------------------------------------
# Fetcher: Race results
# ---------------------------------------------------------------------------

def fetch_race_results(season: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Fetch race finishing order and grid positions from Race session.

    Returns DataFrame with columns:
        season, round, driver_id, driver_name, team, grid_position,
        finish_position, circuit_id
    """
    session = _load_session(season, round_num, "R")
    if session is None:
        return None

    try:
        results = session.results
    except Exception:
        results = None

    if results is None or results.empty:
        logger.warning(f"No race results for {season} R{round_num}")
        return None

    # Extract event info
    circuit_id = session.event["EventName"]

    df = pd.DataFrame(
        {
            "season": season,
            "round": round_num,
            "driver_id": results["Abbreviation"].values,
            "driver_name": results["FullName"].values,
            "team": results["TeamName"].values,
            "grid_position": pd.to_numeric(
                results["GridPosition"], errors="coerce"
            ).values,
            "finish_position": pd.to_numeric(
                results["Position"], errors="coerce"
            ).values,
            "circuit_id": circuit_id,
        }
    )

    logger.info(
        f"Fetched race results: {season} R{round_num} "
        f"({circuit_id}) — {len(df)} drivers"
    )
    return df


# ---------------------------------------------------------------------------
# Fetcher: Qualifying sector times
# ---------------------------------------------------------------------------

def fetch_qualifying(season: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Fetch best qualifying sector times (S1, S2, S3) per driver.

    Includes Correction 8 — missing data guard:
    If >50% of sector times are NaN, logs a warning but does NOT drop or impute.
    Imputation deferred to feature_engineering.py (Week 2).

    Returns DataFrame with columns:
        driver_id, sector_1_time, sector_2_time, sector_3_time
    """
    session = _load_session(season, round_num, "Q")
    if session is None:
        return None

    try:
        laps = session.laps
    except Exception:
        laps = None

    if laps is None or laps.empty:
        logger.warning(f"No qualifying laps for {season} R{round_num}")
        return None

    # -------------------------------------------------------------------
    # Correction 8: Sector time missing data guard
    # -------------------------------------------------------------------
    for sector_col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if sector_col in laps.columns:
            missing_ratio = laps[sector_col].isna().mean()
            if missing_ratio > 0.5:
                logger.warning(
                    f"Season {season} R{round_num}: >50% {sector_col} missing "
                    f"({missing_ratio:.1%}). Flagging — will impute in "
                    f"feature_engineering.py"
                )

    # Get best (fastest) sector times per driver from qualifying
    # Convert timedelta to seconds for numeric processing
    def _td_to_seconds(td_series: pd.Series) -> pd.Series:
        """Convert pandas Timedelta series to float seconds."""
        return td_series.dt.total_seconds()

    # Group by driver, get their fastest lap's sector times
    best_laps = laps.sort_values("LapTime").groupby("Driver").first().reset_index()

    df = pd.DataFrame(
        {
            "driver_id": best_laps["Driver"].values,
            "sector_1_time": _td_to_seconds(best_laps["Sector1Time"]).values
            if "Sector1Time" in best_laps.columns
            else np.nan,
            "sector_2_time": _td_to_seconds(best_laps["Sector2Time"]).values
            if "Sector2Time" in best_laps.columns
            else np.nan,
            "sector_3_time": _td_to_seconds(best_laps["Sector3Time"]).values
            if "Sector3Time" in best_laps.columns
            else np.nan,
        }
    )

    logger.info(
        f"Fetched qualifying sectors: {season} R{round_num} — {len(df)} drivers"
    )
    return df


# ---------------------------------------------------------------------------
# Fetcher: Practice session average lap times
# ---------------------------------------------------------------------------

def fetch_practice(season: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Fetch average lap times from FP2 and FP3 practice sessions.

    Tries FP3 first (closer to race), falls back to FP2.
    Returns mean lap time in seconds per driver.

    Returns DataFrame with columns:
        driver_id, avg_lap_time_practice
    """
    # Try FP3 first, then FP2
    session = _load_session(season, round_num, "FP3")
    if session is None:
        session = _load_session(season, round_num, "FP2")
    if session is None:
        logger.warning(
            f"No practice data available for {season} R{round_num}"
        )
        return None

    try:
        laps = session.laps
    except Exception:
        laps = None

    if laps is None or laps.empty:
        return None

    # Filter out in/out laps and pit laps for cleaner averages
    clean_laps = laps[
        (laps["PitInTime"].isna()) & (laps["PitOutTime"].isna())
    ].copy()

    if clean_laps.empty:
        clean_laps = laps  # Fallback to all laps if filtering too aggressive

    avg_times = (
        clean_laps.groupby("Driver")["LapTime"]
        .mean()
        .reset_index()
    )

    df = pd.DataFrame(
        {
            "driver_id": avg_times["Driver"].values,
            "avg_lap_time_practice": avg_times["LapTime"]
            .dt.total_seconds()
            .values,
        }
    )

    logger.info(
        f"Fetched practice avg laps: {season} R{round_num} — {len(df)} drivers"
    )
    return df


# ---------------------------------------------------------------------------
# Fetcher: Tire data
# ---------------------------------------------------------------------------

def fetch_tire_data(season: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Fetch tire compound, stint lengths, and fresh/used status from Race session.

    Extracts the FIRST stint's data for each driver (race start tire choice).
    Encoding: soft=0, medium=1, hard=2 (intermediate/wet mapped to soft/medium).

    Returns DataFrame with columns:
        driver_id, tire_compound, tire_age_laps, fresh_tire
    """
    session = _load_session(season, round_num, "R")
    if session is None:
        return None

    try:
        laps = session.laps
    except Exception:
        laps = None

    if laps is None or laps.empty:
        return None

    # Tire compound encoding
    compound_map = {
        "SOFT": 0,
        "MEDIUM": 1,
        "HARD": 2,
        "INTERMEDIATE": 1,  # Map wet compounds to nearest dry equivalent
        "WET": 0,
    }

    records = []
    for driver in laps["Driver"].unique():
        driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
        if driver_laps.empty:
            continue

        # Get first stint (starting compound)
        first_lap = driver_laps.iloc[0]
        compound_raw = str(first_lap.get("Compound", "UNKNOWN")).upper()
        compound_encoded = compound_map.get(compound_raw, 1)  # Default medium

        # Tire age: count laps on same compound from start
        first_stint = driver_laps[
            driver_laps["Stint"] == driver_laps["Stint"].iloc[0]
        ]
        tire_age = len(first_stint)

        # Fresh tire: TyreLife at lap 1 — if 0 or 1, it's fresh
        tyre_life = first_lap.get("TyreLife", np.nan)
        if pd.notna(tyre_life):
            fresh = 1 if tyre_life <= 1 else 0
        else:
            fresh = np.nan

        records.append(
            {
                "driver_id": driver,
                "tire_compound": compound_encoded,
                "tire_age_laps": tire_age,
                "fresh_tire": fresh,
            }
        )

    if not records:
        return None

    df = pd.DataFrame(records)
    logger.info(
        f"Fetched tire data: {season} R{round_num} — {len(df)} drivers"
    )
    return df


# ---------------------------------------------------------------------------
# Fetcher: Pit stops (+ team_pit_speed — Correction 3)
# ---------------------------------------------------------------------------

def fetch_pit_stops(season: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Fetch pit stop count and compute team_pit_speed from Race session.

    Correction 3: team_pit_speed computed INSIDE this function as mean pit
    stop duration per team for this race. No separate function needed.

    Returns DataFrame with columns:
        driver_id, team, pit_stop_count, team_pit_speed
    """
    session = _load_session(season, round_num, "R")
    if session is None:
        return None

    try:
        laps = session.laps
    except Exception:
        laps = None

    if laps is None or laps.empty:
        return None

    records = []
    team_pit_durations: dict[str, list[float]] = {}

    for driver in laps["Driver"].unique():
        driver_laps = laps[laps["Driver"] == driver]

        # Count pit stops (transitions between stints)
        stints = driver_laps["Stint"].dropna().unique()
        pit_count = max(0, len(stints) - 1)

        # Get team name from results
        team = "Unknown"
        if session.results is not None and not session.results.empty:
            driver_result = session.results[
                session.results["Abbreviation"] == driver
            ]
            if not driver_result.empty:
                team = driver_result.iloc[0].get("TeamName", "Unknown")

        # Collect pit stop durations for team_pit_speed calculation
        # PitInTime and PitOutTime give us duration
        pit_in_laps = driver_laps[driver_laps["PitInTime"].notna()]
        pit_out_laps = driver_laps[driver_laps["PitOutTime"].notna()]

        # Approximate pit duration from pit in/out times
        for _, pit_lap in pit_in_laps.iterrows():
            pit_in = pit_lap.get("PitInTime")
            pit_out = pit_lap.get("PitOutTime")
            if pd.notna(pit_in) and pd.notna(pit_out):
                duration = (pit_out - pit_in).total_seconds()
                if 0 < duration < 120:  # Sanity check: 0–120 seconds
                    team_pit_durations.setdefault(team, []).append(duration)

        records.append(
            {
                "driver_id": driver,
                "team": team,
                "pit_stop_count": pit_count,
            }
        )

    if not records:
        return None

    df = pd.DataFrame(records)

    # -------------------------------------------------------------------
    # Correction 3: team_pit_speed computed inline
    # -------------------------------------------------------------------
    team_avg_pit = {
        team: np.mean(durations) if durations else np.nan
        for team, durations in team_pit_durations.items()
    }
    df["team_pit_speed"] = df["team"].map(team_avg_pit)

    logger.info(
        f"Fetched pit stops: {season} R{round_num} — {len(df)} drivers, "
        f"{len(team_pit_durations)} teams with pit data"
    )
    return df


# ---------------------------------------------------------------------------
# Fetcher: Weather
# ---------------------------------------------------------------------------

def fetch_weather(season: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Fetch track temperature and rainfall status from Race session weather data.

    Returns DataFrame with columns:
        weather_temp_track, weather_rainfall
    (Single row — same weather for all drivers at race level)
    """
    session = _load_session(season, round_num, "R")
    if session is None:
        return None

    try:
        weather = session.weather_data
    except Exception:
        weather = None

    if weather is None or weather.empty:
        logger.warning(f"No weather data for {season} R{round_num}")
        return None

    # Average track temperature across the race
    track_temp = weather["TrackTemp"].mean() if "TrackTemp" in weather.columns else np.nan

    # Rainfall: 1 if any rain detected during race, 0 otherwise
    rainfall = 0
    if "Rainfall" in weather.columns:
        rainfall = int(weather["Rainfall"].any())

    df = pd.DataFrame(
        {
            "weather_temp_track": [track_temp],
            "weather_rainfall": [rainfall],
        }
    )

    logger.info(
        f"Fetched weather: {season} R{round_num} — "
        f"TrackTemp={track_temp:.1f}°C, Rainfall={bool(rainfall)}"
    )
    return df


# ---------------------------------------------------------------------------
# Fetcher: Lap data for safety car (Correction 2)
# ---------------------------------------------------------------------------

def fetch_lap_data(season: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Fetch lap-by-lap data with track_status for safety car computation.

    Correction 2: Used downstream in merge step to compute
    safety_car_probability per circuit.

    track_status codes:
        '1' = Track clear / Green flag
        '2' = Yellow flag
        '4' = Safety car deployed
        '5' = Red flag
        '6' = Virtual safety car (VSC)

    Returns DataFrame with columns:
        season, round, driver_id, lap_number, track_status, circuit_id
    """
    session = _load_session(season, round_num, "R")
    if session is None:
        return None

    try:
        laps = session.laps
    except Exception:
        laps = None

    if laps is None or laps.empty:
        return None

    circuit_id = session.event["EventName"]

    records = []
    for _, lap in laps.iterrows():
        track_status = str(lap.get("TrackStatus", "1"))
        records.append(
            {
                "season": season,
                "round": round_num,
                "driver_id": lap["Driver"],
                "lap_number": lap.get("LapNumber", np.nan),
                "track_status": track_status,
                "circuit_id": circuit_id,
            }
        )

    if not records:
        return None

    df = pd.DataFrame(records)
    sc_laps = df[df["track_status"] == "4"].shape[0]
    vsc_laps = df[df["track_status"] == "6"].shape[0]

    logger.info(
        f"Fetched lap data: {season} R{round_num} — {len(df)} total laps, "
        f"{sc_laps} SC laps, {vsc_laps} VSC laps"
    )
    return df


# ---------------------------------------------------------------------------
# Season builder: orchestrate all fetchers
# ---------------------------------------------------------------------------

def build_season_dataframe(season: int) -> pd.DataFrame:
    """
    Build a complete DataFrame for one F1 season by orchestrating all fetchers.

    For each round in the season:
      1. Fetch race results (base rows)
      2. Merge qualifying sector times
      3. Merge practice avg lap times
      4. Merge tire data
      5. Merge pit stops (includes team_pit_speed)
      6. Merge weather (broadcast to all drivers)
      7. Fetch lap data (stored separately for safety_car_probability)

    Args:
        season: F1 season year (2018–2026)

    Returns:
        Tuple of (race_df, lap_data_df) — race features and raw lap data
    """
    enable_cache()

    # Get the event schedule for the season
    try:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
    except Exception as e:
        logger.error(f"Could not load {season} schedule: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Filter to only conventional race rounds (exclude pre-season testing etc.)
    # and ONLY include races that have already happened
    from datetime import datetime
    now = pd.to_datetime(datetime.now())
    race_rounds = schedule[
        (schedule["EventFormat"].notna()) & 
        (schedule["EventDate"] < now)
    ]

    all_race_dfs = []
    all_lap_dfs = []

    for _, event in race_rounds.iterrows():
        round_num = event["RoundNumber"]
        if round_num == 0:
            continue  # Skip testing events

        event_name = event.get("EventName", f"R{round_num}")
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {season} R{round_num}: {event_name}")
        logger.info(f"{'='*60}")

        # 1. Race results (base)
        results_df = fetch_race_results(season, round_num)
        if results_df is None:
            logger.warning(f"Skipping {season} R{round_num} — no race results")
            continue

        # 2. Qualifying sector times
        quali_df = fetch_qualifying(season, round_num)
        if quali_df is not None:
            results_df = results_df.merge(
                quali_df, on="driver_id", how="left"
            )
        else:
            results_df["sector_1_time"] = np.nan
            results_df["sector_2_time"] = np.nan
            results_df["sector_3_time"] = np.nan

        # 3. Practice avg lap times
        practice_df = fetch_practice(season, round_num)
        if practice_df is not None:
            results_df = results_df.merge(
                practice_df, on="driver_id", how="left"
            )
        else:
            results_df["avg_lap_time_practice"] = np.nan

        # 4. Tire data
        tire_df = fetch_tire_data(season, round_num)
        if tire_df is not None:
            results_df = results_df.merge(
                tire_df, on="driver_id", how="left"
            )
        else:
            results_df["tire_compound"] = np.nan
            results_df["tire_age_laps"] = np.nan
            results_df["fresh_tire"] = np.nan

        # 5. Pit stops (includes team_pit_speed — Correction 3)
        pit_df = fetch_pit_stops(season, round_num)
        if pit_df is not None:
            # Drop 'team' from pit_df to avoid conflict with results_df
            pit_merge_cols = ["driver_id", "pit_stop_count", "team_pit_speed"]
            results_df = results_df.merge(
                pit_df[pit_merge_cols], on="driver_id", how="left"
            )
        else:
            results_df["pit_stop_count"] = np.nan
            results_df["team_pit_speed"] = np.nan

        # 6. Weather (single row → broadcast to all drivers)
        weather_df = fetch_weather(season, round_num)
        if weather_df is not None:
            results_df["weather_temp_track"] = weather_df[
                "weather_temp_track"
            ].iloc[0]
            results_df["weather_rainfall"] = weather_df[
                "weather_rainfall"
            ].iloc[0]
        else:
            results_df["weather_temp_track"] = np.nan
            results_df["weather_rainfall"] = np.nan

        # Mark all FastF1 rows as having telemetry available
        results_df["telemetry_available"] = True

        all_race_dfs.append(results_df)

        # 7. Lap data (stored separately for safety_car_probability)
        lap_df = fetch_lap_data(season, round_num)
        if lap_df is not None:
            all_lap_dfs.append(lap_df)

    # Combine all rounds
    race_df = (
        pd.concat(all_race_dfs, ignore_index=True) if all_race_dfs else pd.DataFrame()
    )
    lap_data_df = (
        pd.concat(all_lap_dfs, ignore_index=True) if all_lap_dfs else pd.DataFrame()
    )

    if not race_df.empty:
        logger.info(
            f"\n{'='*60}\n"
            f"Season {season} complete: {len(race_df)} rows, "
            f"{race_df['round'].nunique()} races\n"
            f"{'='*60}"
        )

    return race_df, lap_data_df


# ---------------------------------------------------------------------------
# Full dataset builder
# ---------------------------------------------------------------------------

def build_full_dataset(
    start: int = 2018, end: int = 2026
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the complete FastF1 dataset across multiple seasons.

    Args:
        start: First season (default 2018 — FastF1 full coverage starts here)
        end: Last season (inclusive)

    Returns:
        Tuple of (race_df, lap_data_df):
          - race_df: All race features, one row per (season, round, driver)
          - lap_data_df: All lap-by-lap data for safety_car_probability
    """
    enable_cache()

    all_race_dfs = []
    all_lap_dfs = []

    for season in range(start, end + 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"  BUILDING SEASON {season}")
        logger.info(f"{'#'*60}\n")

        race_df, lap_df = build_season_dataframe(season)

        if not race_df.empty:
            all_race_dfs.append(race_df)
        if not lap_df.empty:
            all_lap_dfs.append(lap_df)

    full_race_df = (
        pd.concat(all_race_dfs, ignore_index=True) if all_race_dfs else pd.DataFrame()
    )
    full_lap_df = (
        pd.concat(all_lap_dfs, ignore_index=True) if all_lap_dfs else pd.DataFrame()
    )

    logger.info(
        f"\nFull FastF1 dataset built: {len(full_race_df)} rows across "
        f"{start}–{end}"
    )
    return full_race_df, full_lap_df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KRONECTOR — FastF1 Data Pipeline"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Fetch a single season (e.g., 2023). If not set, fetches 2018–2026.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2018,
        help="Start season for full dataset build (default: 2018)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2026,
        help="End season for full dataset build (default: 2026)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_output",
        help="Directory to save output parquet files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.season:
        # Single season mode (Correction 7: test 2023 first)
        logger.info(f"Fetching single season: {args.season}")
        race_df, lap_df = build_season_dataframe(args.season)

        if not isinstance(race_df, pd.DataFrame) or race_df.empty:
            # build_season_dataframe returns tuple, handle both cases
            if isinstance(race_df, tuple):
                race_df, lap_df = race_df
            else:
                logger.error("No data returned")
                exit(1)
    else:
        # Full dataset mode
        race_df, lap_df = build_full_dataset(args.start, args.end)

    if not race_df.empty:
        race_path = output_dir / "fastf1_races.parquet"
        race_df.to_parquet(race_path, index=False)
        logger.info(f"Saved race data: {race_path} ({len(race_df)} rows)")

        # Print schema summary
        print("\n=== RACE DATA SCHEMA ===")
        print(f"Shape: {race_df.shape}")
        print(f"Columns: {list(race_df.columns)}")
        print(f"\nSample (first 3 rows):")
        print(race_df.head(3).to_string())
        print(f"\nMissing values:")
        print(race_df.isnull().sum().to_string())

    if not lap_df.empty:
        lap_path = output_dir / "fastf1_laps.parquet"
        lap_df.to_parquet(lap_path, index=False)
        logger.info(f"Saved lap data: {lap_path} ({len(lap_df)} rows)")
