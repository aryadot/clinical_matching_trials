"""
pipeline/narrative.py — LLM-Powered Explanations & RAG Chat
Uses Groq (Llama 3.3 70B) to generate clinician-friendly explanations
of why a trial matches a patient, and powers the RAG chat interface.
"""

import os
import json
import streamlit as st


def generate_match_explanation(patient: dict, trial: dict, score_details: dict) -> str:
    """Generate a clinician-readable explanation of why this trial matches this patient."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _fallback_explanation(patient, trial, score_details)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        prompt = f"""You are a clinical trial matching assistant. A patient has been matched to a clinical trial. 
Write a brief (2-3 sentence) clinician-friendly explanation of WHY this trial is relevant for this patient.
Be specific — reference the patient's condition and the trial's focus. Do NOT make eligibility determinations.
Frame as relevance analysis only.

Patient Summary: {patient['raw_summary'][:500]}

Trial: {trial.get('title', '')}
Conditions: {trial.get('conditions', '')}
Interventions: {trial.get('interventions', '')}

Match Score: {score_details.get('match_percentage', 0)}%
Key Reasons: {', '.join(score_details.get('reasons', [])[:4])}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.5,
        )
        return response.choices[0].message.content

    except Exception as e:
        return _fallback_explanation(patient, trial, score_details)


def _fallback_explanation(patient, trial, score_details):
    """Template-based explanation when no API key is available."""
    reasons = score_details.get("reasons", [])
    reasons_text = "; ".join(reasons[:3]) if reasons else "General disease area relevance"
    return (
        f"This trial ({trial.get('title', 'Unknown')[:80]}...) was matched based on: {reasons_text}. "
        f"The composite relevance score is {score_details.get('match_percentage', 0)}%, "
        f"combining semantic similarity, clinical signal matching, and entity overlap analysis."
    )


def chat_with_context(question: str, patient: dict, top_trials: list[dict]) -> str:
    """RAG-grounded chat: answer questions about the patient's trial matches."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Chat requires a Groq API key. Set `GROQ_API_KEY` in your environment."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        # Build context from top matches
        trials_context = ""
        for i, t in enumerate(top_trials[:5], 1):
            trials_context += (
                f"\n{i}. {t['title'][:100]}"
                f"\n   Conditions: {t.get('conditions', 'N/A')}"
                f"\n   Interventions: {t.get('interventions', 'N/A')}"
                f"\n   Match: {t['match_percentage']}%"
                f"\n   Reasons: {', '.join(t.get('reasons', [])[:3])}\n"
            )

        system = f"""You are a clinical trial matching assistant. You have access to a patient profile and their top matched clinical trials.
Answer questions using ONLY this data. Be concise and clinician-friendly.
NEVER make eligibility determinations or medical recommendations. Frame everything as relevance analysis.

Patient Profile:
{patient['raw_summary'][:600]}

Top Matched Trials:
{trials_context}"""

        messages = [{"role": "system", "content": system}]
        if "trial_chat_history" in st.session_state:
            for msg in st.session_state.trial_chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=400,
            temperature=0.5,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"
