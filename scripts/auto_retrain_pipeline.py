"""
KRONECTOR - Auto-Retraining Pipeline
Orchestrates data fetching, drift detection, and MLflow retraining.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from data import run_full_pipeline
from ml.drift_detection import detect_drift
from ml.train import train_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def update_env_file(run_id: str):
    """Updates the KRONECTOR_MODEL_RUN_ID in the .env file."""
    env_path = Path(".env")
    if not env_path.exists():
        logger.warning(".env file not found. Creating a new one.")
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
            
    logger.info("Updated .env with new KRONECTOR_MODEL_RUN_ID")

def main():
    logger.info("=== Starting Auto-Retraining Pipeline ===")
    
    # Define paths
    data_output_dir = Path("data_output")
    data_output_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_output_dir / "fastf1_races.parquet"
    
    # 1. Fetch latest data incrementally
    logger.info("Step 1: Fetching new data incrementally...")
    try:
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            # Find the max FastF1 season we have (telemetry_available == True)
            fastf1_rows = existing_df[existing_df["telemetry_available"] == True]
            if not fastf1_rows.empty:
                last_season = int(fastf1_rows["season"].max())
                start_season = last_season + 1
            else:
                start_season = 2018
                
            if start_season > 2026:
                logger.info("Data is already up to date (2026). Skipping download.")
                df = existing_df
            else:
                logger.info(f"Found existing data up to {last_season}. Resuming download from {start_season}...")
                df = existing_df
                
                # Loop through each missing season one-by-one and save progress!
                for season in range(start_season, 2027):
                    logger.info(f"\n--- PROCESSING SEASON {season} ---")
                    try:
                        new_data = run_full_pipeline(
                            fastf1_start=season,
                            fastf1_end=season,
                            # Dummy years for Jolpica backfill so it skips
                            jolpica_start=season,  
                            jolpica_end=season - 1, 
                        )
                        if not new_data.empty:
                            df = pd.concat([df, new_data], ignore_index=True)
                            df.to_parquet(out_path, index=False)
                            logger.info(f"Successfully appended and SAVED season {season}! Total rows: {len(df)}")
                    except Exception as e:
                        logger.error(f"Failed to process season {season}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue # Continue to next season even if this one fails
        else:
            logger.info("No existing data found. Fetching from scratch...")
            df = run_full_pipeline(
                fastf1_start=2018,
                fastf1_end=2026,
                jolpica_start=2014,
                jolpica_end=2017,
            )
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved initial dataset to {out_path} ({len(df)} rows)")
            
    except Exception as e:
        logger.error(f"Failed to fetch new data: {e}")
        return

    # 2. Run Drift Detection
    logger.info("Step 2: Checking for Data/Concept Drift...")
    # Using 2026 as the current season to evaluate against pre-2026 history
    drift_detected = detect_drift(
        data_path=str(out_path),
        current_season=2026,
        report_output_path="data_output/drift_report.html"
    )
    
    if drift_detected:
        logger.info("Drift detected! Initiating model retraining...")
        # 3. Retrain model
        try:
            run_id = train_model(data_path=str(out_path))
            logger.info(f"Model retrained successfully. New MLflow Run ID: {run_id}")
            
            # 4. Update .env for FastAPI
            update_env_file(run_id)
            logger.info("Auto-retraining pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
    else:
        logger.info("No significant drift detected. Model is still healthy. No retraining needed.")
        logger.info("Auto-retraining pipeline completed successfully.")

if __name__ == "__main__":
    main()
