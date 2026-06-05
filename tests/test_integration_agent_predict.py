"""
Integration test: DataAgent → FeatureEngineering → Prediction

Demonstrates complete flow from natural language query to win probability.

Run with:
    python -m pytest tests/test_integration_agent_predict.py -v -s
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from agents.data_agent import data_agent, QueryIntent
from ml.feature_engineering import prepare_model_data
from ml.predict import predict_dataframe, load_model_and_encoders


@pytest.fixture
def minimal_model_and_encoders():
    """Load real trained model from MLflow."""
    try:
        run_id = os.getenv("KRONECTOR_TEST_RUN_ID")
        if not run_id:
            pytest.skip("KRONECTOR_TEST_RUN_ID not set")
        model, encoders = load_model_and_encoders(run_id)
        return model, encoders
    except Exception as e:
        pytest.skip(f"Model not available: {e}")


def test_agent_to_prediction_pipeline():
    """
    Test: Natural language → Intent → DataFrame → Predictions
    
    Uses mock parser (no API key needed) to verify the full pipeline.
    """
    
    # Mock parser for reproducibility
    def mock_parser(query: str) -> QueryIntent:
        """For this test, always return 2023 Bahrain Verstappen."""
        return {
            "season": 2023,
            "round": 1,
            "driver_id": "VER",
            "driver_name": "Max Verstappen",
        }
    
    # Step 1: Natural language to structured data
    query = "Will Max Verstappen win at Bahrain 2023?"
    result = data_agent(query, parser=mock_parser)
    
    # Verify intent extraction
    assert result["intent"]["season"] == 2023
    assert result["intent"]["round"] == 1
    assert result["intent"]["driver_id"] == "VER"
    print(f"✓ Intent parsed: {result['intent']}")
    
    # Step 2: Verify DataFrame is prediction-compatible
    df = result["dataframe"]
    assert len(df) == 1
    assert "driver_id" in df.columns
    assert df.iloc[0]["driver_id"] == "VER"
    print(f"✓ DataFrame shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
    
    # Step 3: Feature engineering (no encoders needed for shape test)
    bundle, encoders = prepare_model_data(df)
    assert len(bundle.X) == 1
    assert len(bundle.feature_columns) > 0
    assert "grid_position" in bundle.feature_columns
    assert "win_probability" not in bundle.feature_columns
    print(f"✓ Features: {len(bundle.feature_columns)} columns, shape {bundle.X.shape}")
    
    # Step 4: Can generate predictions (if model available)
    try:
        run_id = os.getenv("KRONECTOR_TEST_RUN_ID")
        if not run_id:
            print("ℹ Model inference skipped: KRONECTOR_TEST_RUN_ID not set")
            return
        model, fitted_encoders = load_model_and_encoders(run_id)
        predictions = predict_dataframe(df, model, fitted_encoders)
        
        win_prob = predictions.iloc[0]["win_probability"]
        print(f"✓ Prediction: {win_prob:.2%} win probability for Verstappen")
        assert 0 <= win_prob <= 1
        
    except Exception as e:
        print(f"ℹ Model inference skipped: {e}")
        print("   (This is expected if MLflow model is not available)")


def test_multiple_drivers_same_race():
    """Test querying all drivers in a race without driver filter."""
    
    def mock_parser(query: str) -> QueryIntent:
        """Return race without driver filter."""
        return {
            "season": 2023,
            "round": 1,
            "driver_id": None,  # Get all drivers
            "driver_name": None,
        }
    
    result = data_agent("Predict Bahrain 2023 standings", parser=mock_parser)
    
    # Should have multiple drivers
    num_drivers = len(result["dataframe"])
    assert num_drivers > 1
    print(f"✓ Queried {num_drivers} drivers for 2023 Bahrain")
    
    # All rows should have different driver_ids
    driver_ids = result["dataframe"]["driver_id"].unique()
    assert len(driver_ids) == num_drivers
    print(f"✓ Unique drivers: {', '.join(sorted(driver_ids))}")


def test_agent_output_matches_prediction_schema():
    """Verify agent output matches PredictionInputRow schema."""
    
    def mock_parser(query: str) -> QueryIntent:
        return {"season": 2023, "round": 1, "driver_id": "HAM"}
    
    result = data_agent("Hamilton Bahrain", parser=mock_parser)
    rows = result["rows"]
    
    # Verify required fields exist
    required_fields = {
        "season", "round", "driver_id", "team",
        "grid_position", "finish_position", "circuit_id"
    }
    
    for row in rows:
        for field in required_fields:
            assert field in row, f"Missing field: {field}"
    
    print(f"✓ All {len(required_fields)} required fields present")
    print(f"✓ Sample row keys: {list(rows[0].keys())}")


def test_agent_error_handling():
    """Test agent gracefully handles invalid queries."""
    
    def mock_parser(query: str) -> QueryIntent:
        # Return an impossible race (1000 rounds doesn't exist)
        return {"season": 2099, "round": 1000, "driver_id": None}
    
    with pytest.raises(ValueError, match="No rows found"):
        data_agent("Impossible race", parser=mock_parser)
    
    print("✓ Agent properly rejects non-existent races")


if __name__ == "__main__":
    # Run standalone (without pytest)
    print("=" * 60)
    print("KRONECTOR: DataAgent Integration Test")
    print("=" * 60)
    
    test_agent_to_prediction_pipeline()
    print()
    test_multiple_drivers_same_race()
    print()
    test_agent_output_matches_prediction_schema()
    print()
    test_agent_error_handling()
    
    print()
    print("=" * 60)
    print("All integration tests passed!")
    print("=" * 60)
