"""
KRONECTOR - Compare Agent

Uses Groq's Llama3 to analyze the mathematical SHAP Delta between two drivers
and synthesize a "Tale of the Tape" style narrative.
"""

import os
from groq import Groq

def compare_agent(
    driver1_name: str,
    driver1_prob: float,
    driver1_status: str,
    driver2_name: str,
    driver2_prob: float,
    driver2_status: str,
    shap_deltas: dict[str, float],
    race_context: str
) -> str:
    """Uses Llama3 to write a head-to-head comparison."""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required.")
        
    client = Groq(api_key=api_key)
    
    # Sort deltas to find biggest advantages for driver 1 (positive) and driver 2 (negative)
    sorted_deltas = sorted(shap_deltas.items(), key=lambda item: item[1], reverse=True)
    d1_advantages = [f"{k}: +{v:.2f}" for k, v in sorted_deltas[:3] if v > 0]
    d2_advantages = [f"{k}: +{abs(v):.2f}" for k, v in reversed(sorted_deltas) if v < 0][:3]
    
    system_prompt = f"""
You are KRONECTOR, an elite Formula 1 Race Strategy AI.
You have been asked to compare two drivers head-to-head for an upcoming race.
You have a Machine Learning model that calculates 'SHAP Deltas' — exactly where one driver gains or loses mathematical advantage over the other.

YOUR JOB:
Write a "Tale of the Tape" style boxing-match breakdown (1-2 paragraphs max).
Explain who is the overall favorite based on Win Probability, but highlight the specific areas where the underdog might have an edge.
Make it sound like an expert F1 engineer breaking down a matchup.

IMPORTANT CONCEPTS:
- 'grid_position' or 'pole_conversion_rate' means track position/qualifying pace.
- 'career_race_starts' means veteran experience vs rookie volatility.
- 'driver_form_last3' means recent momentum.
- 'sector_1_time', etc. means raw track speed.
- CRASH DETECTION: You are provided with the 'Qualifying Status' for both drivers. If a driver's status indicates an Accident, Collision, or Crash, you MUST dramatically mention it as a major disadvantage or factor! (Note: a driver can still have a good grid position if they crashed in Q3. If so, highlight this loophole!)
"""

    user_prompt = f"""
RACE CONTEXT: {race_context}

MATCHUP:
{driver1_name} (Win Probability: {driver1_prob:.1f}%) | Quali Status: {driver1_status}
vs
{driver2_name} (Win Probability: {driver2_prob:.1f}%) | Quali Status: {driver2_status}

MATHEMATICAL ADVANTAGES (SHAP Deltas):
Biggest edges for {driver1_name}:
{', '.join(d1_advantages) if d1_advantages else "None"}

Biggest edges for {driver2_name}:
{', '.join(d2_advantages) if d2_advantages else "None"}

Write the Head-to-Head analysis:
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
    
    return chat_completion.choices[0].message.content
