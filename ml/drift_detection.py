"""
KRONECTOR - Drift Detection Pipeline
Monitors incoming F1 data for Concept and Data Drift using Evidently AI.
"""

import logging
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping

from ml.feature_engineering import prepare_model_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def detect_drift(
    data_path: str = "data_output/fastf1_races.parquet",
    current_season: int = 2026,
    report_output_path: str = "data_output/drift_report.html"
) -> bool:
    """
    Detects drift between the reference dataset (historical) and the current dataset.
    Returns True if significant drift is detected.
    """
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        logger.error(f"Data not found at {data_path}")
        return False

    # To detect drift accurately, we should evaluate the model's engineered features.
    # We use prepare_model_data to impute missing values and encode categoricals.
    logger.info("Preparing features for drift detection...")
    bundle, _ = prepare_model_data(df)
    
    # Re-attach season for splitting
    eval_df = bundle.X.copy()
    eval_df["season"] = bundle.metadata["season"]
    eval_df["win_probability"] = bundle.y

    # Define reference (e.g., up to 2025) and current (2026)
    reference = eval_df[eval_df["season"] < current_season].copy()
    current = eval_df[eval_df["season"] >= current_season].copy()
    
    # Drop the season column as it's not a model feature we want to check for drift
    reference = reference.drop(columns=["season"])
    current = current.drop(columns=["season"])

    if current.empty:
        logger.warning(f"No data available for current season {current_season}. Skipping drift detection.")
        return False
        
    logger.info(f"Running drift detection: Reference ({len(reference)} rows) vs Current ({len(current)} rows)")

    # Define column mapping
    target = "win_probability"
    prediction = None
    
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    
    # Since we passed the data through prepare_model_data, all categories are now numerically encoded.
    # We treat them as numerical features for Evidently so it can detect distributional shifts in the encoded space.
    numerical_features = [col for col in reference.columns if col != target]
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = []

    # Create Evidently Report
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])

    logger.info("Generating Evidently AI Drift Report...")
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    
    # Save HTML report
    import os
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    report.save_html(report_output_path)
    logger.info(f"Drift report saved to {report_output_path}")

    # Parse JSON results to determine if drift was detected
    result = report.as_dict()
    
    # Check data drift
    data_drift = False
    target_drift = False
    
    try:
        # Evidently structure navigation
        for metric in result["metrics"]:
            if metric["metric"] == "DataDriftPreset":
                if metric["result"]["dataset_drift"]:
                    data_drift = True
                    logger.warning("🚨 DATA DRIFT DETECTED!")
            elif metric["metric"] == "TargetDriftPreset":
                if metric["result"]["drift_detected"]:
                    target_drift = True
                    logger.warning("🚨 TARGET DRIFT DETECTED!")
    except KeyError as e:
        logger.warning(f"Could not parse drift results exactly: {e}")

    return data_drift or target_drift

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Drift Detection")
    parser.add_argument("--data-path", default="data_output/fastf1_races.parquet")
    parser.add_argument("--current-season", type=int, default=2026)
    args = parser.parse_args()
    
    drift_detected = detect_drift(
        data_path=args.data_path,
        current_season=args.current_season
    )
    print(f"\nFinal Result -> Drift Detected: {drift_detected}")
