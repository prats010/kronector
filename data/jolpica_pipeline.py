"""
KRONECTOR — Jolpica (Ergast fork) Data Pipeline
Backfills F1 data for seasons 2014–2017 from the Jolpica API.
Also provides championship_standing for ALL seasons (2014–2024) — sole source.

Jolpica API: https://api.jolpi.ca/ergast/f1
(Community-maintained fork of deprecated Ergast API, same JSON schema)

Data available:
  - Race results + grid positions
  - Pit stop counts
  - Championship standings (before each race)
  - Circuit metadata

Data NOT available (set to NaN):
  - Sector times, tire data, weather, practice laps
  - telemetry_available = False for all Jolpica-only rows
"""

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from pathlib import Path
import requests
import requests_cache
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# Base URL from .env (default: Jolpica)
JOLPICA_BASE_URL = os.getenv(
    "JOLPICA_BASE_URL", "https://api.jolpi.ca/ergast/f1"
)

# Initialize requests_cache session to prevent rate limits on redundant calls
cache_dir = Path("./cache")
cache_dir.mkdir(parents=True, exist_ok=True)
session = requests_cache.CachedSession(
    str(cache_dir / "jolpica_cache"),
    backend="sqlite",
    expire_after=86400, # 1 day expiration
)


# ---------------------------------------------------------------------------
# Rate-limited request wrapper (Correction 5)
# ---------------------------------------------------------------------------

def jolpica_get(
    url: str, retries: int = 3, base_delay: float = 0.5
) -> Optional[dict[str, Any]]:
    """
    Make a GET request to Jolpica API with exponential backoff.

    Correction 5: 200ms base delay + exponential backoff on failure.
    Used for ALL Jolpica API calls — never use raw requests.get().

    Args:
        url: Full API URL to request
        retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds between requests (default: 0.2)

    Returns:
        Parsed JSON response dict, or None if all retries fail.
    """
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            # Only sleep if it wasn't cached to avoid throttling on local cache
            if not getattr(response, "from_cache", False):
                time.sleep(base_delay)
            return response.json()
        except requests.exceptions.RequestException as e:
            wait = base_delay * (2 ** attempt)
            logger.warning(
                f"Jolpica request failed (attempt {attempt + 1}/{retries}): "
                f"{e}. Retrying in {wait:.1f}s"
            )
            time.sleep(wait)

    logger.error(f"Jolpica request failed after {retries} attempts: {url}")
    return None


# ---------------------------------------------------------------------------
# Fetcher: Race results
# ---------------------------------------------------------------------------

def fetch_race_results(season: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Fetch race results + grid positions from Jolpica API.

    Returns DataFrame with columns:
        season, round, driver_id, driver_name, team, grid_position,
        finish_position, circuit_id
    """
    url = f"{JOLPICA_BASE_URL}/{season}/{round_num}/results.json"
    data = jolpica_get(url)

    if data is None:
        return None

    try:
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            logger.warning(f"No race data for {season} R{round_num}")
            return None

        race = races[0]
        circuit_id = race["Circuit"]["circuitId"]
        results = race["Results"]

        # Import driver map to convert Jolpica slugs → FastF1 abbreviations
        try:
            from data.build_driver_map import DRIVER_MAP

            # Reverse map: jolpica_slug → fastf1_abbreviation
            reverse_map = {v: k for k, v in DRIVER_MAP.items()}
        except ImportError:
            logger.warning(
                "DRIVER_MAP not available — using Jolpica driver IDs as-is"
            )
            reverse_map = {}

        records = []
        for result in results:
            jolpica_id = result["Driver"]["driverId"]
            driver_id = reverse_map.get(jolpica_id, jolpica_id.upper()[:3])

            records.append(
                {
                    "season": season,
                    "round": round_num,
                    "driver_id": driver_id,
                    "driver_name": (
                        f"{result['Driver']['givenName']} "
                        f"{result['Driver']['familyName']}"
                    ),
                    "team": result["Constructor"]["name"],
                    "grid_position": int(result.get("grid", 0)),
                    "finish_position": int(result.get("position", 0)),
                    "circuit_id": circuit_id,
                }
            )

        df = pd.DataFrame(records)
        logger.info(
            f"Fetched Jolpica results: {season} R{round_num} "
            f"({circuit_id}) — {len(df)} drivers"
        )
        return df

    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Failed to parse results for {season} R{round_num}: {e}")
        return None


# ---------------------------------------------------------------------------
# Fetcher: Pit stops
# ---------------------------------------------------------------------------

def fetch_pit_stops(season: int, round_num: int) -> Optional[pd.DataFrame]:
    """
    Fetch pit stop count per driver from Jolpica API.

    Note: Pit stop data available from 2012 onwards in Ergast/Jolpica.
    team_pit_speed NOT computed here (Jolpica doesn't provide duration
    granularity comparable to FastF1). Set to NaN.

    Returns DataFrame with columns:
        driver_id, pit_stop_count
    """
    url = f"{JOLPICA_BASE_URL}/{season}/{round_num}/pitstops.json?limit=100"
    data = jolpica_get(url)

    if data is None:
        return None

    try:
        races = data["MRData"]["RaceTable"]["Races"]
        if not races or "PitStops" not in races[0]:
            return None

        pit_stops = races[0]["PitStops"]

        # Import driver map
        try:
            from data.build_driver_map import DRIVER_MAP

            reverse_map = {v: k for k, v in DRIVER_MAP.items()}
        except ImportError:
            reverse_map = {}

        # Count pit stops per driver
        pit_counts: dict[str, int] = {}
        for stop in pit_stops:
            jolpica_id = stop["driverId"]
            driver_id = reverse_map.get(jolpica_id, jolpica_id.upper()[:3])
            pit_counts[driver_id] = pit_counts.get(driver_id, 0) + 1

        df = pd.DataFrame(
            [
                {"driver_id": did, "pit_stop_count": count}
                for did, count in pit_counts.items()
            ]
        )

        logger.info(
            f"Fetched Jolpica pit stops: {season} R{round_num} — "
            f"{len(df)} drivers"
        )
        return df

    except (KeyError, IndexError, TypeError) as e:
        logger.error(
            f"Failed to parse pit stops for {season} R{round_num}: {e}"
        )
        return None


# ---------------------------------------------------------------------------
# Fetcher: Driver standings (Correction 1 — sole source for ALL years)
# ---------------------------------------------------------------------------

def fetch_driver_standings(
    season: int, round_num: int
) -> Optional[pd.DataFrame]:
    """
    Fetch championship standings BEFORE this race round.

    Correction 1: This is the SOLE source of championship_standing for
    ALL seasons (2014–2024). FastF1 pipeline does NOT fetch this.

    Uses round_num - 1 to get standings BEFORE the current race.
    For round 1, uses previous season's final standings.

    Returns DataFrame with columns:
        driver_id, championship_standing
    """
    # Get standings before this race (previous round's standings)
    if round_num > 1:
        standings_round = round_num - 1
        url = (
            f"{JOLPICA_BASE_URL}/{season}/{standings_round}/"
            f"driverStandings.json"
        )
    else:
        # For round 1, use previous season's final standings
        prev_season = season - 1
        url = f"{JOLPICA_BASE_URL}/{prev_season}/driverStandings.json"

    data = jolpica_get(url)

    if data is None:
        return None

    try:
        standings_lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not standings_lists:
            return None

        standings = standings_lists[0]["DriverStandings"]

        # Import driver map
        try:
            from data.build_driver_map import DRIVER_MAP

            reverse_map = {v: k for k, v in DRIVER_MAP.items()}
        except ImportError:
            reverse_map = {}

        records = []
        for entry in standings:
            jolpica_id = entry["Driver"]["driverId"]
            driver_id = reverse_map.get(jolpica_id, jolpica_id.upper()[:3])
            records.append(
                {
                    "driver_id": driver_id,
                    "championship_standing": int(entry["position"]),
                }
            )

        df = pd.DataFrame(records)
        logger.info(
            f"Fetched standings: {season} R{round_num} (before race) — "
            f"{len(df)} drivers"
        )
        return df

    except (KeyError, IndexError, TypeError) as e:
        logger.error(
            f"Failed to parse standings for {season} R{round_num}: {e}"
        )
        return None


# ---------------------------------------------------------------------------
# Fetcher: Circuit info
# ---------------------------------------------------------------------------

def fetch_circuit_info(
    season: int, round_num: int
) -> Optional[dict[str, str]]:
    """
    Fetch circuit metadata from Jolpica API.

    Returns dict with:
        circuit_id, circuit_name, locality, country
    """
    url = f"{JOLPICA_BASE_URL}/{season}/{round_num}.json"
    data = jolpica_get(url)

    if data is None:
        return None

    try:
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return None

        circuit = races[0]["Circuit"]
        return {
            "circuit_id": circuit["circuitId"],
            "circuit_name": circuit["circuitName"],
            "locality": circuit["Location"]["locality"],
            "country": circuit["Location"]["country"],
        }

    except (KeyError, IndexError, TypeError) as e:
        logger.error(
            f"Failed to parse circuit info for {season} R{round_num}: {e}"
        )
        return None


# ---------------------------------------------------------------------------
# Fetcher: Total rounds in a season
# ---------------------------------------------------------------------------

def fetch_season_rounds(season: int) -> int:
    """Get the total number of race rounds in a season."""
    url = f"{JOLPICA_BASE_URL}/{season}.json"
    data = jolpica_get(url)

    if data is None:
        return 0

    try:
        races = data["MRData"]["RaceTable"]["Races"]
        return len(races)
    except (KeyError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Season builder
# ---------------------------------------------------------------------------

def build_season_dataframe(season: int) -> pd.DataFrame:
    """
    Build a complete Jolpica DataFrame for one F1 season.

    For each round:
      1. Fetch race results (base rows)
      2. Merge pit stop counts
      3. Merge championship standings (before race)
      4. Add NaN columns for telemetry fields (not available from Jolpica)
      5. Set telemetry_available = False

    Args:
        season: F1 season year (2014–2017 for backfill)

    Returns:
        DataFrame with same schema as FastF1 pipeline output
    """
    total_rounds = fetch_season_rounds(season)
    if total_rounds == 0:
        logger.error(f"Could not determine rounds for season {season}")
        return pd.DataFrame()

    logger.info(f"Season {season}: {total_rounds} rounds")

    all_dfs = []

    for round_num in range(1, total_rounds + 1):
        logger.info(f"\nProcessing {season} R{round_num}/{total_rounds}")

        # 1. Race results (base)
        results_df = fetch_race_results(season, round_num)
        if results_df is None:
            logger.warning(f"Skipping {season} R{round_num} — no results")
            continue

        # 2. Pit stops
        pit_df = fetch_pit_stops(season, round_num)
        if pit_df is not None:
            results_df = results_df.merge(
                pit_df, on="driver_id", how="left"
            )
        else:
            results_df["pit_stop_count"] = np.nan

        # 3. Championship standings (Correction 1 — sole source)
        standings_df = fetch_driver_standings(season, round_num)
        if standings_df is not None:
            results_df = results_df.merge(
                standings_df, on="driver_id", how="left"
            )
        else:
            results_df["championship_standing"] = np.nan

        # 4. Add NaN columns for telemetry fields
        # (not available from Jolpica — these come from FastF1 only)
        results_df["sector_1_time"] = np.nan
        results_df["sector_2_time"] = np.nan
        results_df["sector_3_time"] = np.nan
        results_df["avg_lap_time_practice"] = np.nan
        results_df["tire_compound"] = np.nan
        results_df["tire_age_laps"] = np.nan
        results_df["fresh_tire"] = np.nan
        results_df["team_pit_speed"] = np.nan
        results_df["weather_temp_track"] = np.nan
        results_df["weather_rainfall"] = np.nan

        # 5. Mark as no-telemetry
        results_df["telemetry_available"] = False

        all_dfs.append(results_df)

    if not all_dfs:
        return pd.DataFrame()

    season_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        f"\nSeason {season} complete: {len(season_df)} rows, "
        f"{season_df['round'].nunique()} races"
    )
    return season_df


# ---------------------------------------------------------------------------
# Full Jolpica dataset builder (2014–2017 backfill)
# ---------------------------------------------------------------------------

def build_jolpica_dataset(
    start: int = 2014, end: int = 2017
) -> pd.DataFrame:
    """
    Build the complete Jolpica backfill dataset.

    Args:
        start: First season (default 2014)
        end: Last season inclusive (default 2017)

    Returns:
        DataFrame with same schema as FastF1 pipeline output
    """
    all_dfs = []

    for season in range(start, end + 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"  BUILDING JOLPICA SEASON {season}")
        logger.info(f"{'#'*60}\n")

        df = build_season_dataframe(season)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    full_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        f"\nFull Jolpica dataset: {len(full_df)} rows across {start}–{end}"
    )
    return full_df


# ---------------------------------------------------------------------------
# Standalone: Fetch standings for FastF1 years (used in merge step)
# ---------------------------------------------------------------------------

def fetch_all_standings(
    start: int = 2014, end: int = 2026
) -> pd.DataFrame:
    """
    Fetch championship standings for ALL seasons from Jolpica.

    This is the sole source of championship_standing data (Correction 1).
    Called during merge step to join standings onto FastF1 rows.

    Returns DataFrame with columns:
        season, round, driver_id, championship_standing
    """
    all_standings = []

    for season in range(start, end + 1):
        total_rounds = fetch_season_rounds(season)
        for round_num in range(1, total_rounds + 1):
            standings_df = fetch_driver_standings(season, round_num)
            if standings_df is not None:
                standings_df["season"] = season
                standings_df["round"] = round_num
                all_standings.append(standings_df)

    if not all_standings:
        return pd.DataFrame()

    return pd.concat(all_standings, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="KRONECTOR — Jolpica Data Pipeline"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2014,
        help="Start season (default: 2014)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2017,
        help="End season (default: 2017)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_output",
        help="Directory to save output parquet files",
    )
    parser.add_argument(
        "--standings-only",
        action="store_true",
        help="Only fetch standings (for merging with FastF1 data)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.standings_only:
        standings_df = fetch_all_standings(args.start, args.end)
        if not standings_df.empty:
            path = output_dir / "jolpica_standings.parquet"
            standings_df.to_parquet(path, index=False)
            logger.info(f"Saved standings: {path} ({len(standings_df)} rows)")
    else:
        df = build_jolpica_dataset(args.start, args.end)
        if not df.empty:
            path = output_dir / "jolpica_races.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Saved Jolpica data: {path} ({len(df)} rows)")

            print("\n=== JOLPICA DATA SCHEMA ===")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nSample (first 3 rows):")
            print(df.head(3).to_string())
