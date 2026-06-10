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
    critique: CritiqueOutput,
    top_contenders: str = ""
) -> SynthesisOutput:
    """Uses Llama3 to write the final explanation for the user."""
    
    is_prerace = prediction.get("is_prerace", False)
    
    # If mathematically rejected AND it's not a pre-race query → hard reject
    # For pre-race, always call the LLM because the ranking is still meaningful
    if not critique["approved"] and not is_prerace:
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
Explain what the features mean in a natural way. 
Specifically:
- If 'grid_position' or 'pole_conversion_rate' has a strong impact, explain that qualifying position is historically crucial at this specific circuit. Note that our model uses a smoothed historical average, so even newer tracks are grounded in global F1 reality.
- If 'career_race_starts' or 'driver_form_last3' heavily impact a prediction, explain the "experience factor". The model heavily trusts the recent form of proven veterans (100+ starts) but correctly discounts the form of rookies who have a very small sample size of races.
- IMPORTANT: If the user asks about new engine components, fresh power units, or new car parts/upgrades, you MUST acknowledge that our predictive model operates purely on historical telemetry and lap data, and DOES NOT have access to internal team data regarding unannounced parts upgrades or new engines. 
- You have been provided the top contenders for this race. If the predicted driver is NOT one of the top favorites, you MUST mention the actual favorites to provide accurate context (e.g., "While Antonelli has a 5% chance, keep in mind Verstappen and Hamilton are the heavy favorites").
- CRASH DETECTION: You will be provided the driver's 'Qualifying Status'. If it says something like 'Accident', 'Collision', 'Spun off', or anything indicating a crash, you MUST dramatically mention that they crashed in qualifying. Note that a driver can still have a good grid position (e.g. they set a fast lap in Q3 and THEN crashed, which is a known loophole). If they crashed but have a good grid position, explicitly point out this loophole!

IMPORTANT — If Confidence Rating is "PreRace":
This is a PRE-RACE prediction using only qualifying data. Tire strategy, weather, and pit-stop data are not yet available.
The model's absolute probabilities are compressed across 22 drivers, so the RANKING matters more than the exact number.
Frame your response as a pre-race assessment, not a definitive prediction. Mention the key qualifying factors (sector times, grid position, driver form).

Keep it concise (1-2 paragraphs).
"""

    # For pre-race rejected predictions (below 5% threshold), explain it's a long-shot
    prerace_note = ""
    if not critique["approved"] and is_prerace:
        prerace_note = f"\nNOTE: This driver's win probability ({prediction['probability']:.1%}) is very low — they are considered a long-shot for this race based on qualifying data."

    user_prompt = f"""
USER QUERY:
"{query}"

PREDICTED DRIVER: {prediction.get('driver_name', 'Unknown Driver')}
Qualifying Status: {prediction.get('quali_status', 'Finished')}

ML PREDICTION:
Win Probability: {prediction['probability']:.1%}
Confidence Rating: {critique['confidence_rating']}
{prerace_note}

OVERALL RACE FAVORITES:
{top_contenders if top_contenders else "Unknown"}

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
