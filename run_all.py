import os
from pathlib import Path
from data import run_full_pipeline
from ml.train import train_model

def main():
    print("Starting full data pipeline (2014-2024)...")
    print("This may take some time as it fetches and caches historical telemetry.")
    
    df = run_full_pipeline(
        fastf1_start=2018,
    fastf1_end=2026,
        jolpica_start=2014,
        jolpica_end=2017,
    )
    
    output_dir = Path("data_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fastf1_races.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved full dataset to {out_path} ({len(df)} rows)")
    
    print("\nTraining LightGBM model on full dataset...")
    run_id = train_model(data_path=str(out_path))
    print(f"Model trained successfully! MLflow Run ID: {run_id}")
    
    # Update .env
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            for line in lines:
                if line.startswith("KRONECTOR_MODEL_RUN_ID="):
                    f.write(f"KRONECTOR_MODEL_RUN_ID={run_id}\n")
                else:
                    f.write(line)
        print("Updated .env with new KRONECTOR_MODEL_RUN_ID")

if __name__ == "__main__":
    main()
