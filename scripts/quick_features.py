import pandas as pd
import numpy as np

print("Loading existing dataset...")
df = pd.read_parquet('data_output/fastf1_races.parquet')

print("1. Calculating Bayesian pole_conversion_rate...")
if 'pole_conversion_rate' in df.columns:
    df = df.drop(columns=['pole_conversion_rate'])

completed = df.dropna(subset=['finish_position']).copy()
poles = completed[completed['grid_position'] == 1].copy()
poles['pole_won'] = (poles['finish_position'] == 1).astype(int)

global_mean = poles['pole_won'].mean()
C = 3.0
stats = poles.groupby('circuit_id').agg(
    wins=('pole_won', 'sum'),
    total=('pole_won', 'count')
).reset_index()

stats['pole_conversion_rate'] = (stats['wins'] + C * global_mean) / (stats['total'] + C)
df = df.merge(stats[['circuit_id', 'pole_conversion_rate']], on='circuit_id', how='left')
df['pole_conversion_rate'] = df['pole_conversion_rate'].fillna(global_mean)

print("2. Calculating career_race_starts...")
if 'career_race_starts' in df.columns:
    df = df.drop(columns=['career_race_starts'])

df_sorted = df.sort_values(by=['driver_id', 'season', 'round']).copy()
df_sorted['career_race_starts'] = df_sorted.groupby('driver_id').cumcount()
df = df.merge(df_sorted[['season', 'round', 'driver_id', 'career_race_starts']], on=['season', 'round', 'driver_id'], how='left')
df['career_race_starts'] = df['career_race_starts'].fillna(0)

print(f"Saving {len(df)} rows...")
df.to_parquet('data_output/fastf1_races.parquet', index=False)
print("Done!")
