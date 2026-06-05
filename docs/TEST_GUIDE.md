# Complete Step-by-Step Testing Guide

## Part 1: Check Available Models

### Step 1: List all MLflow runs

Open a terminal and run:

```bash
python -c "
import mlflow
runs = mlflow.search_runs()
if runs:
    for i, run in enumerate(runs):
        print(f'{i+1}. Run ID: {run.info.run_id}')
        print(f'   Status: {run.info.status}')
        artifacts = mlflow.artifacts.list_artifacts(run_id=run.info.run_id)
        print(f'   Has artifacts: {len(artifacts) > 0}')
        print()
else:
    print('No runs found!')
"
```

**What to look for:**
- Copy a run ID that has `status: FINISHED`
- Check if it has artifacts (models)

**If you see run IDs:** Go to Step 3  
**If you see "No runs found!":** Go to Step 2

---

## Part 2: Train a New Model (If Needed)

### Step 2: Train the model

```bash
python ml/train.py
```

**What to expect:**
- Takes 2-5 minutes
- Prints: `Training complete! Run ID: ...`
- Copy that Run ID

---

## Part 3: Update Configuration

### Step 3: Update .env with the correct Run ID

Edit `.env` and replace the run ID:

```bash
KRONECTOR_MODEL_RUN_ID=<paste-the-run-id-here>
```

Example:
```
KRONECTOR_MODEL_RUN_ID=563634279953604033
```

**Save the file!**

### Step 4: Restart the API

If the API is running, stop it (Ctrl+C in the terminal).

Then restart:

```bash
python -m uvicorn api.main:app --reload
```

Wait for: `Uvicorn running on http://127.0.0.1:8000`

---

## Part 4: Test the API

### Step 5: Check Health Status

Open a new terminal and run:

```bash
curl http://localhost:8000/health
```

**Expected output:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "data_available": true,
  "version": "1.0.0"
}
```

**If `model_loaded` is still `false`:** The run ID is wrong. Go back to Step 1 and copy the correct ID.

### Step 6: List Available Drivers

```bash
curl "http://localhost:8000/drivers?season=2023"
```

**Expected output:**
```json
[
  {"driver_id": "VER", "driver_name": "Max Verstappen", "team": "Red Bull"},
  {"driver_id": "SAI", "driver_name": "Carlos Sainz", "team": "Ferrari"},
  ...
]
```

### Step 7: List Races

```bash
curl http://localhost:8000/races/2023
```

**Expected output:**
```json
[
  {"season": 2023, "round": 1, "name": "Bahrain Grand Prix"},
  {"season": 2023, "round": 2, "name": "Saudi Arabian Grand Prix"},
  ...
]
```

---

## Part 5: Test the Main Prediction

### Step 8: Natural Language Query (THE MAIN TEST)

```bash
curl -X POST http://localhost:8000/predict/f1 `
  -H "Content-Type: application/json" `
  -d '{"query": "What is Verstappens win probability at Monaco 2023?"}'
```

**Expected output:**
```json
{
  "win_probability": 0.87,
  "metadata": {
    "season": 2023,
    "round": 6,
    "driver_id": "VER",
    "driver_name": "Max Verstappen",
    "team": "Red Bull",
    "grid_position": 1
  },
  "shap_values": {
    "grid_position": 0.45,
    "previous_race_points": 0.23,
    ...
  }
}
```

**If it works:** ✅ SUCCESS! The system is working!

---

## Part 6: Visual Testing (Easiest!)

### Step 9: Use the Interactive API Docs

Open your browser and visit:

```
http://localhost:8000/docs
```

**You'll see all 5 endpoints with:**
- Try it out buttons
- Example inputs
- Full responses

**Test order:**
1. Click `/health` → Execute (check model_loaded)
2. Click `/drivers` → Execute (see drivers)
3. Click `/races/{season}` → Enter "2023" → Execute
4. Click `/predict/f1` → Enter `{"query": "Verstappen Monaco 2023"}` → Execute

---

## Troubleshooting

### Problem: `model_loaded: false`

**Solution:**
1. Check you have the right run ID in `.env`
2. Verify the run exists: `python ml/train.py` to create a new one
3. Restart the API

### Problem: Query returns error 503

**Solution:**
- Model not loaded yet (see above)
- Check the API logs in the terminal for details

### Problem: `"query": "must be at least 3 characters"`

**Solution:**
- Make sure your query is longer than 3 characters
- Example: `"What is VER's win probability at Monaco?"` ✅

### Problem: `Race not found`

**Solution:**
- Check valid seasons: 2023 races are available
- Try: `"Verstappen Bahrain 2023"` instead

---

## Quick Test Commands (Copy & Paste)

```bash
# 1. Check health
curl http://localhost:8000/health

# 2. List drivers
curl "http://localhost:8000/drivers?season=2023"

# 3. List races
curl http://localhost:8000/races/2023

# 4. Make prediction
curl -X POST http://localhost:8000/predict/f1 `
  -H "Content-Type: application/json" `
  -d '{"query": "Verstappen Monaco 2023 win probability?"}'

# 5. Open interactive docs
start http://localhost:8000/docs
```

---

## Expected Results

| Test | Status | Output |
|------|--------|--------|
| Health check | ✅ | `model_loaded: true` |
| List drivers | ✅ | Array of drivers |
| List races | ✅ | Array of races |
| Prediction | ✅ | Win probability (0-1) + SHAP values |

---

## Success!

If all 4 tests pass, your system is fully functional! 🎉

The API is ready to:
- Accept natural language F1 queries
- Return accurate win probabilities
- Explain predictions with SHAP values
