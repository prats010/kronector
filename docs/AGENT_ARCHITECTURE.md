# KRONECTOR DataAgent Architecture

## Overview

The **DataAgent** (`agents/data_agent.py`) transforms natural-language F1 queries into prediction-ready DataFrames compatible with the ML pipeline. It bridges user intent with model-ready data.

```
User Query → Groq LLM (parse) → Extract Intent → Query Data → DataFrame → predict_dataframe()
```

## Components

### 1. **Query Intent Extraction** 

Function: `parse_query_with_groq(query: str) -> QueryIntent`

Uses **Groq llama3-70b-8192** to parse natural language and extract:
- `season` (int): F1 season year
- `round` (int): Race round number  
- `driver_id` (str | None): 3-letter FastF1 driver code
- `driver_name` (str | None): Driver full name
- `query_intent` (str | None): What the user is asking

**Prompt Strategy:**
- Strict JSON-only output (zero temperature)
- Uses FastF1 3-letter driver codes (VER, HAM, etc.)
- Fails safely with clear error messages

**Example:**
```python
intent = parse_query_with_groq("What was Max's win probability at Monaco 2023?")
# Returns: {"season": 2023, "round": 6, "driver_id": "VER", "driver_name": "Max Verstappen"}
```

### 2. **Data Filtering & Assembly**

Functions:
- `_load_feature_data(path)` - Load parquet dataset
- `_filter_by_intent(df, intent)` - Filter by season/round/driver
- `build_prediction_dataframe(intent, data_path)` - Complete filtering pipeline

**Input Schema (from fastf1_pipeline.py):**
```
season, round, driver_id, driver_name, team
grid_position, finish_position, circuit_id
sector_1_time, sector_2_time, sector_3_time
avg_lap_time_practice
tire_compound, tire_age_laps, fresh_tire
pit_stop_count, team_pit_speed
weather_temp_track, weather_rainfall
championship_standing
```

### 3. **Data Pipeline Integration**

The returned DataFrame flows directly into:

```python
# DataAgent output
result = data_agent("Verstappen Monaco 2023")
df = result["dataframe"]

# Compatible with ML pipeline
bundle, encoders = prepare_model_data(df)
predictions = predict_dataframe(df, model, encoders)
```

**No intermediate transformations needed** — the agent returns FastF1-schema data that feature_engineering.py handles.

## API Reference

### Main Entry Point

```python
def data_agent(
    query: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    parser: IntentParser | None = None,
) -> DataAgentOutput:
    """
    Parse natural-language query and return prediction-ready rows.
    
    Args:
        query: Natural-language F1 query
        data_path: Path to fastf1_races.parquet
        parser: Optional custom parser (for testing/offline use)
    
    Returns:
        DataAgentOutput with:
        - query (str): Original user query
        - intent (QueryIntent): Parsed season/round/driver
        - rows (list[PredictionInputRow]): Matching race rows as dicts
        - dataframe (pd.DataFrame): Full DataFrame for ML pipeline
    """
```

### CLI Usage

```bash
# Parse and display results
python -m agents.data_agent "Verstappen Monaco 2023"

# Output as JSON
python -m agents.data_agent "Hamilton Silverstone 2023" --json

# Custom data path
python -m agents.data_agent "Norris Austin 2024" --data-path /path/to/races.parquet
```

## Testing

**Run tests:**
```bash
python -m pytest tests/test_data_agent.py -v
```

**Test fixtures include:**
- Groq API mocking for unit tests
- Temporary parquet datasets  
- Prediction pipeline compatibility checks

## Error Handling

| Error | Cause | Resolution |
|-------|-------|-----------|
| `GROQ_API_KEY is required` | Missing env var | Set `GROQ_API_KEY` in `.env` |
| `No rows found for season=X round=Y` | Race doesn't exist in data | Check calendar/data availability |
| `No rows matched driver intent` | Driver not in race | Verify driver code and race round |
| `JSON parse error` | Groq returned invalid JSON | Retry with lower temperature |

## Design Decisions

### Plain Python (No LangGraph)
- Groq API is lightweight and doesn't require orchestration framework
- Single synchronous call for query parsing
- Future: LangGraph can wrap this for multi-turn conversations

### TypedDict Schema
- Ensures type safety across data pipeline
- Explicit columns prevent silent failures
- Compatible with mypy strict mode

### Groq for Intent Extraction
- Fast: 70B model with 8K context window
- Accurate: Instruction-tuned for structured output
- Cost: $0.19/$0.39 per million tokens (vs Claude)

## Example Workflows

### Single Race Prediction
```python
from agents.data_agent import data_agent
from ml.predict import load_model_and_encoders, predict_dataframe

agent = DataAgent()
result = agent.query("Will Verstappen win Monza 2023?")

model, encoders = load_model_and_encoders(run_id="abc123")
predictions = predict_dataframe(result["dataframe"], model, encoders)

print(predictions[["driver_id", "driver_name", "win_probability"]])
```

### Batch Queries (future)
```python
queries = [
    "Verstappen Monaco 2023",
    "Hamilton Silverstone 2023", 
    "Norris Spa 2023",
]

for q in queries:
    result = data_agent(q)
    pred = predict_dataframe(result["dataframe"], model, encoders)
    print(f"{q}: {pred.iloc[0]['win_probability']:.2%}")
```

### Testing with Mock Parser
```python
from agents.data_agent import data_agent, QueryIntent

def mock_parser(query: str) -> QueryIntent:
    return {
        "season": 2023,
        "round": 1, 
        "driver_id": "VER",
    }

result = data_agent("test query", parser=mock_parser)
# Works without GROQ_API_KEY
```

## Future Enhancements

1. **LangGraph Integration** — Multi-turn clarification ("Which Hamilton race?")
2. **Result Summarization** — "Max has 87% win probability at Monaco"
3. **Model Explanations** — Attach SHAP values to predictions
4. **Context Awareness** — Current standings, form, tire strategy
5. **Caching** — Avoid re-parsing identical queries

## Files

- **Main:** [agents/data_agent.py](agents/data_agent.py)
- **Tests:** [tests/test_data_agent.py](tests/test_data_agent.py)
- **Upstream (Input Schema):** [data/fastf1_pipeline.py](data/fastf1_pipeline.py)
- **Downstream (ML Pipeline):** [ml/feature_engineering.py](ml/feature_engineering.py), [ml/predict.py](ml/predict.py)
