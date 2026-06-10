"""
KRONECTOR — Pre-Race Prediction Data Builder
=============================================
Builds prediction rows for an UPCOMING race using:
  - Qualifying sector times + grid positions (from FastF1 qualifying session)
  - Practice avg lap times (FP2/FP3)
  - Driver form from existing historical parquet
  - Championship standings from existing historical parquet

This allows predictions BEFORE the race happens, using only qualifying data.

Usage:
  python -m scripts.build_prerace_rows --season 2026 --round 7 --output data_output/prerace_monaco_2026.parquet
  python -m scripts.build_prerace_rows --season 2026 --round 7  # saves to data_output/fastf1_races.parquet as temp rows
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fastf1_pipeline import enable_cache, fetch_qualifying, fetch_practice

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def fetch_grid_from_qualifying(season: int, round_num: int) -> pd.DataFrame | None:
    """
    Load the qualifying session and extract grid positions + sector times.
    Uses lightweight load (laps only, no telemetry/weather/messages) so it works
    even when data was just published.

    Returns DataFrame with columns: driver_id, driver_name, team, grid_position, circuit_id
    """
    enable_cache()
    try:
        session = fastf1.get_session(season, round_num, "Q")
        # Only load laps — skip telemetry, weather, messages (not needed + cause errors on fresh data)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        logger.info(f"Qualifying session loaded for {season} R{round_num}")
    except Exception as e:
        logger.error(f"Could not load qualifying session for {season} R{round_num}: {e}")
        logger.error("💡 If qualifying just finished, wait ~15-30 minutes for FastF1 to process the data, then try again.")
        return None

    # Try results table first (gives clean classification order)
    try:
        results = session.results
    except Exception:
        results = None

    if results is not None and not results.empty:
        circuit_id = session.event["EventName"]
        df = pd.DataFrame({
            "driver_id": results["Abbreviation"].values,
            "driver_name": results["FullName"].values,
            "team": results["TeamName"].values,
            "grid_position": range(1, len(results) + 1),
            "circuit_id": circuit_id,
            "quali_status": results["Status"].values if "Status" in results.columns else "Finished",
        })
        logger.info(f"✅ Grid from qualifying results: {season} R{round_num} ({circuit_id}) — {len(df)} drivers")
        logger.info(f"   P1: {df.iloc[0]['driver_name']} ({df.iloc[0]['team']})")
        return df

    # Fallback: derive grid from laps (best lap time ordering)
    try:
        laps = session.laps
    except Exception:
        laps = None

    if laps is None or laps.empty:
        logger.warning(f"No qualifying laps available for {season} R{round_num}")
        logger.warning("💡 Data may not be available yet — try again in 15-30 minutes.")
        return None

    circuit_id = session.event["EventName"]

    # Best lap per driver → sort ascending to get grid order
    best = laps.groupby("Driver")["LapTime"].min().sort_values().reset_index()
    # Map driver abbreviation to full name/team via session results if available
    driver_info = {}
    if session.results is not None and not session.results.empty:
        for _, r in session.results.iterrows():
            driver_info[r["Abbreviation"]] = {
                "driver_name": r.get("FullName", r["Abbreviation"]),
                "team": r.get("TeamName", "Unknown"),
            }

    records = []
    for pos, (_, row) in enumerate(best.iterrows(), start=1):
        abbr = row["Driver"]
        info = driver_info.get(abbr, {"driver_name": abbr, "team": "Unknown"})
        records.append({
            "driver_id": abbr,
            "driver_name": info["driver_name"],
            "team": info["team"],
            "grid_position": pos,
            "circuit_id": circuit_id,
            "quali_status": "Finished",
        })

    df = pd.DataFrame(records)
    logger.info(f"✅ Grid from lap times (fallback): {season} R{round_num} ({circuit_id}) — {len(df)} drivers")
    logger.info(f"   P1: {df.iloc[0]['driver_name']} ({df.iloc[0]['team']})")
    return df


def get_driver_context(existing_df: pd.DataFrame, season: int, round_num: int) -> pd.DataFrame:
    """
    Pull driver_form_last3 and championship_standing for each driver
    from the most recent data available before this round.
    """
    # Get data up to but NOT including this round/season
    prior = existing_df[
        (existing_df["season"] < season) |
        ((existing_df["season"] == season) & (existing_df["round"] < round_num))
    ].copy()

    if prior.empty:
        logger.warning("No prior data found for driver context — using defaults")
        return pd.DataFrame(columns=["driver_id", "driver_form_last3", "championship_standing"])

    # Latest championship standing per driver this season (or prior season)
    context_rows = []
    for driver_id in prior["driver_id"].unique():
        driver_data = prior[prior["driver_id"] == driver_id].sort_values(
            ["season", "round"], ascending=True
        )
        if driver_data.empty:
            continue

        # driver_form_last3: mean finish position of last 3 races
        recent = driver_data.tail(3)
        form = recent["finish_position"].mean() if "finish_position" in recent.columns else np.nan

        # championship_standing: most recent value
        standing = np.nan
        if "championship_standing" in driver_data.columns:
            valid = driver_data["championship_standing"].dropna()
            if not valid.empty:
                standing = valid.iloc[-1]

        # safety_car_probability: get from the circuit if available
        context_rows.append({
            "driver_id": driver_id,
            "driver_form_last3": form,
            "championship_standing": standing,
        })

    return pd.DataFrame(context_rows)


def build_prerace_rows(
    season: int,
    round_num: int,
    existing_parquet: str = "data_output/fastf1_races.parquet",
) -> pd.DataFrame | None:
    """
    Build a DataFrame of pre-race prediction rows for an upcoming race.

    These rows use qualifying + practice data for features, and set:
      - finish_position = NaN (unknown — will be imputed/excluded)
      - win_probability target = 0 (dummy — not used in inference)
      - telemetry_available = True

    Returns a DataFrame compatible with ml.predict.predict_dataframe()
    """
    # 1. Get grid from qualifying results
    grid_df = fetch_grid_from_qualifying(season, round_num)
    if grid_df is None:
        return None

    # 2. Get qualifying sector times
    quali_sectors = fetch_qualifying(season, round_num)
    if quali_sectors is not None:
        grid_df = grid_df.merge(quali_sectors, on="driver_id", how="left")
    else:
        logger.warning("No qualifying sector times — using NaN")
        grid_df["sector_1_time"] = np.nan
        grid_df["sector_2_time"] = np.nan
        grid_df["sector_3_time"] = np.nan

    # 3. Get practice lap times
    practice_df = fetch_practice(season, round_num)
    if practice_df is not None:
        grid_df = grid_df.merge(practice_df, on="driver_id", how="left")
    else:
        grid_df["avg_lap_time_practice"] = np.nan

    # 4. Pull driver form + championship standing from existing historical data
    if Path(existing_parquet).exists():
        existing_df = pd.read_parquet(existing_parquet)
        context_df = get_driver_context(existing_df, season, round_num)
        grid_df = grid_df.merge(context_df, on="driver_id", how="left")
    else:
        logger.warning(f"No existing parquet at {existing_parquet} — driver context will be NaN")
        grid_df["driver_form_last3"] = np.nan
        grid_df["championship_standing"] = np.nan

    # 5. Fill in race-level fields that won't be known until after the race
    grid_df["season"] = season
    grid_df["round"] = round_num
    grid_df["finish_position"] = np.nan     # Unknown — will be imputed
    grid_df["tire_compound"] = np.nan
    grid_df["tire_age_laps"] = np.nan
    grid_df["fresh_tire"] = np.nan
    grid_df["pit_stop_count"] = np.nan
    grid_df["team_pit_speed"] = np.nan
    grid_df["weather_temp_track"] = np.nan
    grid_df["weather_rainfall"] = np.nan
    grid_df["telemetry_available"] = True

    # safety_car_probability: use historical average for this circuit if available
    if Path(existing_parquet).exists():
        existing_df = pd.read_parquet(existing_parquet)
        circuit_name = grid_df["circuit_id"].iloc[0]
        circuit_data = existing_df[
            existing_df["circuit_id"].astype(str).str.contains(
                circuit_name.split()[0], case=False, na=False
            )
        ]
        if not circuit_data.empty and "safety_car_probability" in circuit_data.columns:
            sc_prob = circuit_data["safety_car_probability"].dropna().mean()
            grid_df["safety_car_probability"] = sc_prob if not np.isnan(sc_prob) else 0.0
        else:
            grid_df["safety_car_probability"] = 0.0

        # pole_conversion_rate: historical pole-to-win % at this circuit
        if not circuit_data.empty and "finish_position" in circuit_data.columns:
            poles = circuit_data[circuit_data["grid_position"] == 1].dropna(subset=["finish_position"])
            if not poles.empty:
                pcr = (poles["finish_position"] == 1).mean()
                grid_df["pole_conversion_rate"] = pcr
                logger.info(f"   Pole conversion rate at {circuit_name}: {pcr:.1%}")
            else:
                grid_df["pole_conversion_rate"] = 0.5
        else:
            grid_df["pole_conversion_rate"] = 0.5
            
        # career_race_starts: count driver's past races
        for idx, row in grid_df.iterrows():
            d_id = row["driver_id"]
            starts = len(existing_df[existing_df["driver_id"] == d_id])
            grid_df.at[idx, "career_race_starts"] = starts
            
    else:
        grid_df["safety_car_probability"] = 0.0
        grid_df["pole_conversion_rate"] = 0.5
        grid_df["career_race_starts"] = 0

    logger.info(f"\n✅ Pre-race rows built for {season} R{round_num} — {len(grid_df)} drivers")
    logger.info(f"   Grid P1: {grid_df.iloc[0]['driver_name']} (qualifying pole)")
    logger.info("\nGrid order:")
    for _, row in grid_df.iterrows():
        s1 = f"{row.get('sector_1_time', np.nan):.3f}s" if pd.notna(row.get('sector_1_time')) else "N/A"
        logger.info(f"  P{int(row['grid_position'])}: {row['driver_name']} ({row['team']}) | S1={s1}")

    return grid_df


def main():
    parser = argparse.ArgumentParser(description="KRONECTOR — Pre-Race Row Builder")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_num")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Defaults to data_output/prerace_{circuit}_{season}.parquet",
    )
    parser.add_argument(
        "--existing",
        type=str,
        default="data_output/fastf1_races.parquet",
        help="Existing historical parquet for driver form/standings context",
    )
    args = parser.parse_args()

    df = build_prerace_rows(
        season=args.season,
        round_num=args.round_num,
        existing_parquet=args.existing,
    )

    if df is None:
        logger.error("Failed to build pre-race rows. Exiting.")
        sys.exit(1)

    # Save output — default to data_output/prerace/ so the API auto-loads it on startup
    circuit_slug = df["circuit_id"].iloc[0].replace(" ", "_").lower()[:20] if not df.empty else "unknown"
    out_path = args.output or f"data_output/prerace/prerace_{circuit_slug}_{args.season}.parquet"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    logger.info(f"\n💾 Saved to: {out_path}")
    print(f"\nPre-race parquet saved: {out_path}")
    print(f"Rows: {len(df)} drivers")
    print(f"\nTo run predictions:")
    print(f"  Use the API: POST /predict/f1 with data_path pointing to this file")
    print(f"  Or: python -m ml.predict --run-id <YOUR_RUN_ID> --data-path {out_path}")


if __name__ == "__main__":
    main()
