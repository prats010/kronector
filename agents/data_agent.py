"""
KRONECTOR - DataAgent.

Plain Python agent that turns a natural-language F1 query into a DataFrame
compatible with ml.predict.predict_dataframe().

Usage:
    agent = DataAgent()
    df = agent.query("What was Verstappen's win probability at Monaco 2023?")
    # Returns a DataFrame with one row matching the query
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, NotRequired, TypedDict

import pandas as pd
from groq import Groq


DEFAULT_DATA_PATH = Path("data_output/fastf1_races.parquet")
# Current available model (as of June 2026)
GROQ_MODEL = "llama-3.3-70b-versatile"


class QueryIntent(TypedDict):
    season: int
    round: NotRequired[int | None]
    grand_prix: NotRequired[str | None]
    driver_id: NotRequired[str | None]
    driver_name: NotRequired[str | None]
    query_intent: NotRequired[str]


class PredictionInputRow(TypedDict, total=False):
    season: int
    round: int
    driver_id: str
    driver_name: str
    team: str
    grid_position: float
    finish_position: float
    circuit_id: str
    sector_1_time: float
    sector_2_time: float
    sector_3_time: float
    avg_lap_time_practice: float
    tire_compound: float
    tire_age_laps: float
    fresh_tire: float
    pit_stop_count: float
    team_pit_speed: float
    weather_temp_track: float
    weather_rainfall: float
    championship_standing: float
    driver_form_last3: float
    safety_car_probability: float
    telemetry_available: bool
    regulation_era: str
    track_type: str
    win_probability: int


class DataAgentOutput(TypedDict):
    query: str
    intent: QueryIntent
    rows: list[PredictionInputRow]
    dataframe: pd.DataFrame


IntentParser = Callable[[str], QueryIntent]


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Groq response did not contain JSON: {text!r}")
    return json.loads(match.group(0))


def _coerce_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required query field: {field_name}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc


def _normalize_driver_id(value: Any) -> str | None:
    if value is None:
        return None
    driver_id = str(value).strip()
    if not driver_id:
        return None
    return driver_id.upper()


def parse_query_with_groq(query: str) -> QueryIntent:
    """
    Use Groq llama3-70b-8192 to extract season, round, and driver intent.

    Expected JSON shape:
        {"season": 2023, "round": 1, "driver_id": "VER", "driver_name": "Max Verstappen"}
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for DataAgent query parsing")

    client = Groq(api_key=api_key)
    prompt = (
        "Extract F1 prediction intent from the user query. Return only JSON with "
        "keys season (int), round (int, or null if the user asks by race name), "
        "grand_prix (str, e.g. 'Canadian Grand Prix', or null), "
        "driver_id, driver_name. Use a FastF1 three-letter "
        "driver_id when clear, otherwise null. Do not add prose.\n\n"
        f"Query: {query}"
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON parser for Formula 1 race queries.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    parsed = _extract_json_object(content)
    
    # Season is required
    season = _coerce_int(parsed.get("season"), "season")
    
    # We need either round or grand_prix
    round_num = parsed.get("round")
    if round_num is not None:
        round_num = _coerce_int(round_num, "round")
        
    return {
        "season": season,
        "round": round_num,
        "grand_prix": parsed.get("grand_prix"),
        "driver_id": _normalize_driver_id(parsed.get("driver_id")),
        "driver_name": parsed.get("driver_name"),
    }


def _load_feature_data(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature data not found: {path}")
    return pd.read_parquet(path)


def _filter_by_intent(df: pd.DataFrame, intent: QueryIntent) -> pd.DataFrame:
    rows = df[df["season"] == intent["season"]].copy()
    
    round_num = intent.get("round")
    grand_prix = intent.get("grand_prix")
    
    if round_num is not None:
        rows = rows[rows["round"] == round_num]
    elif grand_prix is not None:
        name_lower = str(grand_prix).strip().lower()
        rows = rows[
            rows["circuit_id"].astype(str).str.lower().str.contains(
                re.escape(name_lower), na=False
            )
        ]
        
        # If we matched a grand prix string, but it happened multiple times in the season 
        # (extremely rare, e.g. 2020 Austrian/Styrian), just take the first round that matched.
        if not rows.empty:
            matched_round = rows["round"].iloc[0]
            rows = rows[rows["round"] == matched_round]
    else:
        raise ValueError("Either 'round' or 'grand_prix' must be provided in the query.")

    if rows.empty:
        raise ValueError(
            f"No rows found for season={intent['season']} round={round_num} grand_prix={grand_prix}"
        )

    driver_id = _normalize_driver_id(intent.get("driver_id"))
    driver_name = intent.get("driver_name")

    if driver_id:
        rows = rows[rows["driver_id"].astype(str).str.upper() == driver_id]
    elif driver_name:
        name_lower = str(driver_name).strip().lower()
        rows = rows[
            rows["driver_name"].astype(str).str.lower().str.contains(
                re.escape(name_lower), na=False
            )
        ]

    if rows.empty:
        raise ValueError(f"No rows matched driver intent: {intent}")

    return rows.reset_index(drop=True)


def build_prediction_dataframe(intent: QueryIntent, data_path: str | Path) -> pd.DataFrame:
    """Build a predict_dataframe-compatible DataFrame from parsed intent."""
    df = _load_feature_data(data_path)
    return _filter_by_intent(df, intent)


def data_agent(
    query: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    parser: IntentParser | None = None,
) -> DataAgentOutput:
    """
    Parse a natural-language query and return prediction-ready rows.

    Args:
        query: Natural-language F1 query.
        data_path: Parquet dataset with fastf1_pipeline/feature schema.
        parser: Optional parser override for tests or offline use.
    """
    parse = parser or parse_query_with_groq
    intent = parse(query)
    dataframe = build_prediction_dataframe(intent, data_path)

    return {
        "query": query,
        "intent": intent,
        "rows": dataframe.to_dict(orient="records"),
        "dataframe": dataframe,
    }


def main() -> None:
    """CLI entry point for the data agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="KRONECTOR DataAgent: Natural language to F1 prediction data"
    )
    parser.add_argument("query", help="Natural language F1 query")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help=f"Path to feature dataset (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output rows as JSON instead of DataFrame repr",
    )
    args = parser.parse_args()

    try:
        result = data_agent(args.query, data_path=args.data_path)
        print(f"Query: {result['query']}")
        print(f"Parsed Intent: {json.dumps(result['intent'], indent=2)}")
        print(f"Rows found: {len(result['rows'])}\n")

        if args.json:
            print(json.dumps(result["rows"], indent=2))
        else:
            print(result["dataframe"].to_string(index=False))

    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        raise


if __name__ == "__main__":
    main()
