import fastf1
from data.fastf1_pipeline import enable_cache

enable_cache()
session = fastf1.get_session(2026, 1, 'Q')
session.load(laps=True, telemetry=False, weather=False, messages=False)

if session.results is not None:
    print(session.results[['Abbreviation', 'Status']].head(22))
