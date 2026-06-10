import fastf1
from data.fastf1_pipeline import enable_cache
enable_cache()
s = fastf1.get_session(2021, 'Monaco', 'Q')
s.load(laps=True, telemetry=False, weather=False, messages=False)
print(s.results.columns)
print(s.results[s.results['Abbreviation'] == 'LEC'])
