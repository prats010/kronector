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
    is_prerace = prediction.get("is_prerace", False)
    
    approved = True
    confidence_rating = "Normal"
    critique_notes = ""
    
    # 1. Evaluate Probability Thresholds
    # Pre-race predictions use a lower threshold (5% = true random chance in a 22-car field)
    # because features like tires/weather are missing and probabilities are naturally spread thin.
    # Post-race/historical queries use the stricter 20% threshold.
    reject_threshold = 0.05 if is_prerace else 0.20
    
    if prob < reject_threshold:
        approved = False
        confidence_rating = "Low"
        context = "pre-race estimate" if is_prerace else "historical query"
        critique_notes += f"REJECTED: Win probability is {prob:.1%}, below the {reject_threshold:.0%} threshold for a {context}. The model is not confident.\n\n"
    elif is_prerace and prob < 0.20:
        confidence_rating = "PreRace"
        critique_notes += f"PRE-RACE ESTIMATE: Win probability is {prob:.1%}. This is a pre-race prediction using qualifying data only — tires, weather, and race pace data are unavailable. Rankings are meaningful but absolute probabilities are lower than post-race queries.\n\n"
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
