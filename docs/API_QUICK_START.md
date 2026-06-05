# KRONECTOR API Quick Start

## Installation

Requirements already in `requirements.txt`:
- fastapi>=0.109.0
- uvicorn[standard]>=0.27.0
- pydantic (included with fastapi)

No additional installs needed!

---

## Configuration

Set these environment variables in `.env`:

```bash
GROQ_API_KEY=your-groq-key  # For natural language parsing
KRONECTOR_MODEL_RUN_ID=abc123  # MLflow run ID for trained model
```

Optional:
```bash
KRONECTOR_TEST_RUN_ID=abc123  # For integration tests
```

---

## Start the Server

```bash
python -m uvicorn api.main:app --reload
```

Or with custom host/port:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Output:**
```
INFO:     Application startup complete
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## API Documentation

### Interactive Docs (Swagger UI)
Visit: **http://localhost:8000/docs**

### ReDoc
Visit: **http://localhost:8000/redoc**

---

## Endpoints

### 1. Predict Race Outcome

**Endpoint:**
```
POST /predict/f1
```

**Request:**
```json
{
  "query": "What's Max Verstappen's win probability at Monaco 2023?"
}
```

**Response:**
```json
{
  "win_probability": 0.87,
  "metadata": {
    "season": 2023,
    "round": 6,
    "driver_id": "VER",
    "driver_name": "Max Verstappen",
    "team": "Red Bull Racing",
    "grid_position": 1.0
  },
  "shap_values": {
    "grid_position": 0.45,
    "sector_1_time": 0.12,
    "team_pit_speed": -0.05
  }
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/predict/f1 \
  -H "Content-Type: application/json" \
  -d '{"query": "Verstappen Monaco 2023"}'
```

---

### 2. List Drivers

**Endpoint:**
```
GET /drivers
GET /drivers?season=2023
```

**Response:**
```json
[
  {
    "driver_id": "VER",
    "driver_name": "Max Verstappen",
    "team": "Red Bull Racing"
  },
  {
    "driver_id": "HAM",
    "driver_name": "Lewis Hamilton",
    "team": "Mercedes"
  }
]
```

**cURL:**
```bash
curl http://localhost:8000/drivers
curl http://localhost:8000/drivers?season=2023
```

---

### 3. List Races

**Endpoint:**
```
GET /races/{season}
```

**Response:**
```json
[
  {
    "season": 2023,
    "round": 1,
    "name": "Bahrain Grand Prix"
  },
  {
    "season": 2023,
    "round": 2,
    "name": "Saudi Arabian Grand Prix"
  }
]
```

**cURL:**
```bash
curl http://localhost:8000/races/2023
```

---

### 4. Health Check

**Endpoint:**
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "data_available": true,
  "version": "1.0.0"
}
```

**cURL:**
```bash
curl http://localhost:8000/health
```

---

## Error Responses

### 400 Bad Request
Invalid query or missing race data.

```json
{
  "detail": "No matching data for query. Season 2099, round 999"
}
```

### 503 Service Unavailable
Model or data not loaded.

```json
{
  "detail": "Model not loaded. Set KRONECTOR_MODEL_RUN_ID."
}
```

### 422 Unprocessable Entity
Validation error (e.g., query too short).

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "ensure this value has at least 3 characters",
      "type": "value_error.string.min_length"
    }
  ]
}
```

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Predict win probability
response = requests.post(
    f"{BASE_URL}/predict/f1",
    json={"query": "What's Lewis' chance at Silverstone 2023?"}
)
prediction = response.json()
print(f"Win probability: {prediction['win_probability']:.1%}")
print(f"Driver: {prediction['metadata']['driver_name']}")

# List drivers
drivers = requests.get(f"{BASE_URL}/drivers").json()
print(f"Total drivers: {len(drivers)}")

# List races
races = requests.get(f"{BASE_URL}/races/2023").json()
print(f"Races in 2023: {len(races)}")

# Health check
health = requests.get(f"{BASE_URL}/health").json()
print(f"API Status: {health['status']}")
```

---

## Running Tests

```bash
# All API tests
python -m pytest tests/test_api_endpoints.py -v

# With output
python -m pytest tests/test_api_endpoints.py -v -s

# Specific test
python -m pytest tests/test_api_endpoints.py::test_health_endpoint -v
```

---

## Performance Notes

- **First prediction**: ~2-3 seconds (model inference)
- **Subsequent predictions**: ~1 second (cached model)
- **Driver/race list**: <100ms
- **Health check**: <10ms

---

## Troubleshooting

### "Model not loaded"
- Set `KRONECTOR_MODEL_RUN_ID` environment variable
- Verify MLflow run exists: `mlflow runs list --experiment-id 0`

### "Race data not loaded"
- Verify `data_output/fastf1_races.parquet` exists
- Run data pipeline first: `python -m data.fastf1_pipeline`

### "Query parsing failed"
- Set `GROQ_API_KEY` environment variable
- Query must be 3+ characters

### Port already in use
```bash
python -m uvicorn api.main:app --port 8001
```

---

## Files

- [api/main.py](../api/main.py) — FastAPI application
- [api/schemas.py](../api/schemas.py) — Pydantic models
- [tests/test_api_endpoints.py](../tests/test_api_endpoints.py) — Tests

---

**Status:** ✅ Production-ready API

Ready to predict! 🏁
