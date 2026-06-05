import requests
import json

print('=' * 70)
print('TEST: Natural Language Prediction (Multiple Formats)')
print('=' * 70)

test_queries = [
    'Verstappen Bahrain 2023 win probability?',
    'Max Verstappen Bahrain Grand Prix 2023',
    'Round 1 2023 Verstappen',
    'VER Bahrain 2023',
]

for query in test_queries:
    print('\nTrying: ' + query)
    try:
        resp = requests.post(
            'http://127.0.0.1:8000/predict/f1',
            json={'query': query},
            timeout=30
        )
        result = resp.json()
        if 'win_probability' in result:
            print('  ✓ SUCCESS!')
            print('  Win probability: {}'.format(result.get('win_probability')))
            print('  Driver: {} ({})'.format(
                result.get('metadata', {}).get('driver_name'),
                result.get('metadata', {}).get('driver_id')
            ))
        else:
            print('  ✗ Error: {}'.format(result.get('detail')))
    except Exception as e:
        print('  ✗ Exception: {}'.format(str(e)))
