"""
KRONECTOR - Critique Agent

Evaluates the mathematical probability and SHAP values produced by the PredictionAgent.
Rejects weak predictions, flags overconfidence, and extracts the top driving features.
"""

from typing import TypedDict
from agents.prediction_agent import PredictionOutput

class CritiqueOutput(TypedDict):
    approved: bool
    confidence_rating: str
    critique_notes: str

def critique_agent(prediction: PredictionOutput) -> CritiqueOutput:
    """Critiques the prediction mathematically to safeguard the LLM."""
    prob = prediction["probability"]
    shap_values = prediction["shap_values"]
    
    approved = True
    confidence_rating = "Normal"
    critique_notes = ""
    
    # 1. Evaluate Probability Thresholds
    if prob < 0.20:
        approved = False
        confidence_rating = "Low"
        critique_notes += f"REJECTED: Win probability is {prob:.1%}, which is below the 20% threshold. The model is guessing. Do not make a definitive prediction.\n\n"
    elif prob > 0.95:
        confidence_rating = "Overconfident"
        critique_notes += f"FLAGGED: Win probability is unusually high ({prob:.1%}). In modern F1, >95% confidence may indicate data leakage or a hyper-dominant driver in the dataset. Proceed with caution.\n\n"
    else:
        critique_notes += f"APPROVED: Win probability is {prob:.1%}, which is mathematically sound for a confident prediction.\n\n"
        
    # 2. Extract SHAP Driving Features
    if shap_values:
        # Sort features by absolute SHAP value magnitude
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        top_3 = sorted_features[:3]
        
        critique_notes += "TOP MATHEMATICAL DRIVING FACTORS (SHAP Analysis):\n"
        for feature, value in top_3:
            impact = "Positive impact" if value > 0 else "Negative impact"
            critique_notes += f"- {feature}: {impact} (SHAP value: {value:.4f})\n"
            
    return {
        "approved": approved,
        "confidence_rating": confidence_rating,
        "critique_notes": critique_notes.strip()
    }
