"""
KRONECTOR - Synthesis Agent

Uses Groq's Llama3 to synthesize the mathematical critique and user query 
into a natural, intelligent F1 race-engineer style response.
"""

import os
from typing import TypedDict
from groq import Groq

from agents.prediction_agent import PredictionOutput
from agents.critique_agent import CritiqueOutput

class SynthesisOutput(TypedDict):
    final_response: str

def synthesis_agent(
    query: str, 
    prediction: PredictionOutput, 
    critique: CritiqueOutput
) -> SynthesisOutput:
    """Uses Llama3 to write the final explanation for the user."""
    
    # If the mathematical critique rejected the prediction, bypass the LLM
    if not critique["approved"]:
        return {
            "final_response": (
                "I apologize, but my underlying machine learning model is not confident enough "
                f"to make a prediction on this (Probability: {prediction['probability']:.1%}). "
                "I have mathematically rejected this query to prevent hallucination."
            )
        }
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required.")
        
    client = Groq(api_key=api_key)
    
    system_prompt = """
You are KRONECTOR, an elite Formula 1 Race Strategy AI.
You have been asked a question by a user.
You have a LightGBM Machine Learning model that just ran the numbers to predict the probability of a driver winning.
Another mathematical agent (the CritiqueAgent) has analyzed the SHAP values to tell you exactly WHICH features drove this prediction.

YOUR JOB:
Synthesize this mathematical data into a clear, confident, and professional answer for the user.
Speak like an F1 Race Engineer on the pit wall.
Do NOT hallucinate reasons. Only use the features provided in the Critique Notes.
Explain what the features mean in a natural way (e.g. if 'grid_position' has a positive impact, say starting on pole is a massive advantage here).

Keep it concise (1-2 paragraphs).
"""

    user_prompt = f"""
USER QUERY:
"{query}"

PREDICTED DRIVER: {prediction.get('driver_name', 'Unknown Driver')}

ML PREDICTION:
Win Probability: {prediction['probability']:.1%}
Confidence Rating: {critique['confidence_rating']}

CRITIQUE AGENT NOTES (SHAP ANALYSIS):
{critique['critique_notes']}

Write the final response:
"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=500,
    )
    
    return {
        "final_response": chat_completion.choices[0].message.content
    }
