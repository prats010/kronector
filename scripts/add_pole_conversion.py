"""One-off script to add pole_conversion_rate to the existing historical parquet."""
import pandas as pd
import numpy as np

df = pd.read_parquet("data_output/fastf1_races.parquet")
print(f"Loaded {len(df)} rows")

# Compute pole conversion rate per circuit
completed = df.dropna(subset=["finish_position"]).copy()
poles = completed[completed["grid_position"] == 1].copy()
poles["pole_won"] = (poles["finish_position"] == 1).astype(int)
pcr = (
    poles.groupby("circuit_id")["pole_won"]
    .mean()
    .reset_index()
    .rename(columns={"pole_won": "pole_conversion_rate"})
)

print("\nPole conversion rates:")
for _, row in pcr.sort_values("pole_conversion_rate", ascending=False).iterrows():
    circuit = row["circuit_id"]
    rate = row["pole_conversion_rate"]
    print(f"  {circuit:30s} {rate:.1%}")

# Merge onto main df
if "pole_conversion_rate" in df.columns:
    df = df.drop(columns=["pole_conversion_rate"])
df = df.merge(pcr, on="circuit_id", how="left")
df["pole_conversion_rate"] = df["pole_conversion_rate"].fillna(0.5)

df.to_parquet("data_output/fastf1_races.parquet", index=False)
print(f"\nSaved with pole_conversion_rate — {len(df)} rows")
