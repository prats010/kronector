"""
KRONECTOR - Agent Chain Test Script
"""

import os
from dotenv import load_dotenv

from agents.prediction_agent import PredictionOutput
from agents.critique_agent import critique_agent
from agents.synthesis_agent import synthesis_agent

def main():
    load_dotenv()
    
    # Simulate a user query
    query = "Why did the model predict Max Verstappen to win the 2026 Monaco GP?"
    
    # Simulate a PredictionOutput from LightGBM
    mock_prediction: PredictionOutput = {
        "probability": 0.82,
        "shap_values": {
            "grid_position": -0.8500, # Starting pole (lower grid pos is better, so SHAP might show negative impact on position but positive on win. Let's just use positive numbers for impact)
            "driver_form_last3": 0.6200,
            "team_pit_speed": 0.3100,
            "safety_car_probability": -0.1500,
            "weather_rainfall": 0.0500
        },
        "feature_names": [
            "grid_position", "driver_form_last3", "team_pit_speed", 
            "safety_car_probability", "weather_rainfall"
        ],
        "model_version": "latest",
        "run_id": "test_123"
    }
    
    print("--- 1. CRITIQUE AGENT ---")
    critique = critique_agent(mock_prediction)
    print(f"Approved: {critique['approved']}")
    print(f"Confidence: {critique['confidence_rating']}")
    print("Notes:")
    print(critique["critique_notes"])
    print("\n")
    
    print("--- 2. SYNTHESIS AGENT (Llama3) ---")
    synthesis = synthesis_agent(query, mock_prediction, critique)
    print(synthesis["final_response"])

if __name__ == "__main__":
    main()
