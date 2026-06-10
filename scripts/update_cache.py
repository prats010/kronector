"""
KRONECTOR — Incremental Cache & Data Update Script
====================================================
Run this after each new race weekend to:
  1. Fetch data for specific rounds that are NEW (not already in the parquet).
  2. Append them to data_output/fastf1_races.parquet.
  3. Retrain the model on the full updated dataset.
  4. Update .env with the new MLflow run_id.

Usage:
  # Fetch rounds 6, 7 for 2026 and retrain:
  python -m scripts.update_cache --season 2026 --rounds 6 7

  # Fetch and append only, skip retraining:
  python -m scripts.update_cache --season 2026 --rounds 6 7 --no-retrain
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1

import builtins
import requests
import urllib3
builtins.RequestsCookieJar = requests.cookies.RequestsCookieJar
builtins.HTTPAdapter = requests.adapters.HTTPAdapter
builtins.Retry = urllib3.util.Retry

# Add project root so we can import ml.train etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fastf1_pipeline import (
    enable_cache,
    fetch_race_results,
    fetch_qualifying,
    fetch_practice,
    fetch_tire_data,
    fetch_pit_stops,
    fetch_weather,
    fetch_lap_data,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def fetch_single_round(season: int, round_num: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Fetch all data for a single race round, mirroring build_season_dataframe logic.
    Returns (race_df, lap_df). Either can be None if the session isn't available yet.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Fetching {season} Round {round_num}")
    logger.info(f"{'='*60}")

    # 1. Race results (base) — if race hasn't happened, this will be None
    results_df = fetch_race_results(season, round_num)
    if results_df is None:
        logger.warning(f"⚠️  No race results for {season} R{round_num} — race may not have happened yet.")
        return None, None

    # 2. Qualifying sector times
    quali_df = fetch_qualifying(season, round_num)
    if quali_df is not None:
        results_df = results_df.merge(quali_df, on="driver_id", how="left")
    else:
        results_df["sector_1_time"] = np.nan
        results_df["sector_2_time"] = np.nan
        results_df["sector_3_time"] = np.nan

    # 3. Practice avg lap times
    practice_df = fetch_practice(season, round_num)
    if practice_df is not None:
        results_df = results_df.merge(practice_df, on="driver_id", how="left")
    else:
        results_df["avg_lap_time_practice"] = np.nan

    # 4. Tire data
    tire_df = fetch_tire_data(season, round_num)
    if tire_df is not None:
        results_df = results_df.merge(tire_df, on="driver_id", how="left")
    else:
        results_df["tire_compound"] = np.nan
        results_df["tire_age_laps"] = np.nan
        results_df["fresh_tire"] = np.nan

    # 5. Pit stops
    pit_df = fetch_pit_stops(season, round_num)
    if pit_df is not None:
        pit_merge_cols = ["driver_id", "pit_stop_count", "team_pit_speed"]
        results_df = results_df.merge(pit_df[pit_merge_cols], on="driver_id", how="left")
    else:
        results_df["pit_stop_count"] = np.nan
        results_df["team_pit_speed"] = np.nan

    # 6. Weather
    weather_df = fetch_weather(season, round_num)
    if weather_df is not None:
        results_df["weather_temp_track"] = weather_df["weather_temp_track"].iloc[0]
        results_df["weather_rainfall"] = weather_df["weather_rainfall"].iloc[0]
    else:
        results_df["weather_temp_track"] = np.nan
        results_df["weather_rainfall"] = np.nan

    results_df["telemetry_available"] = True

    # 7. Lap data for safety car
    lap_df = fetch_lap_data(season, round_num)

    logger.info(f"✅  Round {round_num} fetched: {len(results_df)} driver rows")
    return results_df, lap_df


def update_env_file(run_id: str):
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(f"KRONECTOR_MODEL_RUN_ID={run_id}\n")
        return

    with open(env_path, "r") as f:
        lines = f.readlines()

    found = False
    with open(env_path, "w") as f:
        for line in lines:
            if line.startswith("KRONECTOR_MODEL_RUN_ID="):
                f.write(f"KRONECTOR_MODEL_RUN_ID={run_id}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"KRONECTOR_MODEL_RUN_ID={run_id}\n")

    logger.info(f"✅  Updated .env with KRONECTOR_MODEL_RUN_ID={run_id}")


def main():
    parser = argparse.ArgumentParser(description="KRONECTOR — Incremental Cache Updater")
    parser.add_argument("--season", type=int, required=True, help="F1 season year (e.g. 2026)")
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        required=True,
        help="Round numbers to fetch (e.g. --rounds 6 7 8)",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip model retraining after data update (just update the parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_output/fastf1_races.parquet",
        help="Path to the parquet file to update",
    )
    args = parser.parse_args()

    enable_cache()

    out_path = Path(args.output)

    # ----------------------------------------------------------------
    # Load existing data
    # ----------------------------------------------------------------
    if out_path.exists():
        logger.info(f"Loading existing dataset from {out_path}...")
        existing_df = pd.read_parquet(out_path)
        logger.info(f"  Existing rows: {len(existing_df)}")

        # Check which rounds already exist for this season
        existing_rounds = set(
            existing_df[existing_df["season"] == args.season]["round"].dropna().astype(int).unique()
        )
        logger.info(f"  Rounds already in dataset for {args.season}: {sorted(existing_rounds)}")
    else:
        logger.error(f"No existing parquet found at {out_path}. Run run_all.py first.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Fetch each requested round
    # ----------------------------------------------------------------
    new_race_dfs = []

    for round_num in args.rounds:
        if round_num in existing_rounds:
            logger.info(f"⏭️  Round {round_num} already exists in dataset — skipping. "
                        f"(Use --force to overwrite)")
            continue

        race_df, _ = fetch_single_round(args.season, round_num)
        if race_df is not None and not race_df.empty:
            new_race_dfs.append(race_df)

    if not new_race_dfs:
        logger.info("No new rounds were fetched. Dataset unchanged.")
        if not args.no_retrain:
            logger.info("Skipping retraining since no data changed.")
        sys.exit(0)

    # ----------------------------------------------------------------
    # Append and save
    # ----------------------------------------------------------------
    combined_df = pd.concat([existing_df] + new_race_dfs, ignore_index=True)
    combined_df.to_parquet(out_path, index=False)

    added = len(combined_df) - len(existing_df)
    logger.info(f"\n✅  Dataset updated: {len(existing_df)} → {len(combined_df)} rows (+{added} new rows)")
    logger.info(f"   Saved to {out_path}")

    # ----------------------------------------------------------------
    # Retrain
    # ----------------------------------------------------------------
    if args.no_retrain:
        logger.info("Skipping retraining (--no-retrain flag set).")
        return

    logger.info("\n🔧 Starting model retraining on updated dataset...")
    try:
        from ml.train import train_model
        run_id = train_model(data_path=str(out_path))
        logger.info(f"\n🏁 Model retrained! New MLflow Run ID: {run_id}")
        update_env_file(run_id)
        print(f"\n{'='*60}")
        print(f"  NEW KRONECTOR_MODEL_RUN_ID = {run_id}")
        print(f"  Copy this to your Hugging Face Space secrets!")
        print(f"{'='*60}\n")
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
