import requests
import json

# Test 1: Health check
print('=' * 70)
print('TEST 1: Health Check')
print('=' * 70)
resp = requests.get('http://127.0.0.1:8000/health')
print(json.dumps(resp.json(), indent=2))
print()

# Test 2: List drivers
print('=' * 70)
print('TEST 2: List Drivers (2023)')
print('=' * 70)
resp = requests.get('http://127.0.0.1:8000/drivers?season=2023')
drivers = resp.json()
print('Found {} drivers'.format(len(drivers)))
print('First 3:')
for d in drivers[:3]:
    print('  {}'.format(d))
print()

# Test 3: List races
print('=' * 70)
print('TEST 3: List Races (2023)')
print('=' * 70)
resp = requests.get('http://127.0.0.1:8000/races/2023')
races = resp.json()
print('Found {} races'.format(len(races)))
print('First 3:')
for r in races[:3]:
    print('  Round {}: {}'.format(r.get('round'), r.get('name')))
print()

# Test 4: MAIN TEST
print('=' * 70)
print('TEST 4: Natural Language Prediction')
print('=' * 70)
query = 'What is Max Verstappens win probability at Monaco 2023?'
print('Query: {}'.format(query))
resp = requests.post(
    'http://127.0.0.1:8000/predict/f1',
    json={'query': query},
    timeout=30
)
result = resp.json()
print(json.dumps(result, indent=2))
