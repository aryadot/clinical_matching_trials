"""
pipeline/eligibility.py — Criterion-Level Eligibility Checker
==============================================================
Checks a patient profile against each inclusion criterion of a trial
using Groq LLM. Returns per-criterion assessment and an overall summary.

Design principles:
  - One LLM call per trial (not one per criterion) for efficiency
  - Outputs are framed as decision support — never autonomous determinations
  - All outputs include "discuss with your oncologist" framing
  - HIPAA-aware: no real patient data stored or logged

Usage:
    from pipeline.eligibility import check_eligibility
    result = check_eligibility(patient, trial, groq_api_key)
"""

import json
import os

ELIGIBILITY_PROMPT = """You are a clinical oncology assistant helping patients understand which clinical trials may be relevant to discuss with their oncologist.

Patient Profile:
{patient_summary}

Patient Details:
- Age: {age}
- Receptor Status: {receptor_status}
- Disease Stage: {stage}
- Prior Treatments: {prior_treatments}
- Other Notes: {comorbidities}

Trial: {trial_title}
Conditions: {trial_conditions}

Inclusion Criteria to Review:
{criteria_list}

For each numbered criterion, assess whether this patient appears to meet it based on the available information.
Use MEETS, DOES NOT MEET, or UNCLEAR (if information is missing or ambiguous).

Important: This is for informational purposes only. Final eligibility requires physician review.

Respond in this exact JSON format:
{{
  "criteria_assessment": [
    {{"criterion": "<criterion text>", "status": "MEETS|DOES NOT MEET|UNCLEAR", "note": "<one brief reason>"}}
  ],
  "overall_summary": "<2-3 sentence plain language summary for the patient>",
  "recommend_discussion": true
}}"""


def _parse_inclusion_criteria(eligibility_text: str) -> list[str]:
    """Extract inclusion criteria lines from eligibility text."""
    if not eligibility_text:
        return []

    lower = eligibility_text.lower()
    inc_idx = lower.find("inclusion criteria")
    exc_idx = lower.find("exclusion criteria")

    if inc_idx != -1 and exc_idx != -1:
        block = eligibility_text[inc_idx:exc_idx]
    elif inc_idx != -1:
        block = eligibility_text[inc_idx:]
    else:
        block = eligibility_text[:1000]

    criteria = []
    for line in block.splitlines():
        line = line.strip().lstrip("*-•·123456789. ")
        if len(line) > 20 and "inclusion criteria" not in line.lower():
            criteria.append(line)

    return criteria[:10]  # Cap at 10 to keep LLM call efficient


def check_eligibility(patient: dict, trial: dict, groq_api_key: str) -> dict:
    """
    Check patient against trial inclusion criteria using Groq LLM.

    Args:
        patient: Parsed patient dict with raw_summary, age, receptor_status etc.
        trial: Trial dict with title, conditions, eligibility text
        groq_api_key: Groq API key

    Returns:
        dict with criteria_assessment list, overall_summary, and recommend_discussion flag
        Returns empty result gracefully on failure
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        criteria = _parse_inclusion_criteria(trial.get("eligibility", ""))
        if not criteria:
            return {
                "criteria_assessment": [],
                "overall_summary": "Eligibility criteria not available for this trial. Please check ClinicalTrials.gov directly.",
                "recommend_discussion": True
            }

        # Format criteria as numbered list
        criteria_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))

        # Format patient details
        rs = patient.get("receptor_status", {})
        receptor_str = ", ".join(
            f"{k}: {v}" for k, v in rs.items() if v != "unknown"
        ) or "Unknown"

        prior_tx = patient.get("prior_treatments", [])
        prior_tx_str = ", ".join(prior_tx) if prior_tx else "None reported"

        comorbidities = patient.get("comorbidities", [])
        comorbidities_str = ", ".join(comorbidities) if comorbidities else "None reported"

        prompt = ELIGIBILITY_PROMPT.format(
            patient_summary=patient.get("raw_summary", "")[:400],
            age=patient.get("age", "Unknown"),
            receptor_status=receptor_str,
            stage=patient.get("stage", "Unknown"),
            prior_treatments=prior_tx_str,
            comorbidities=comorbidities_str,
            trial_title=trial.get("title", "Unknown Trial")[:100],
            trial_conditions=trial.get("conditions", "N/A")[:100],
            criteria_list=criteria_list,
        )

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "criteria_assessment": result.get("criteria_assessment", []),
            "overall_summary": result.get("overall_summary", ""),
            "recommend_discussion": True
        }

    except Exception as e:
        return {
            "criteria_assessment": [],
            "overall_summary": "Could not complete eligibility check. Please review this trial with your oncologist.",
            "recommend_discussion": True,
            "error": str(e)
        }


def build_patient_from_form(
    age: int,
    diagnosis: str,
    her2: str,
    er: str,
    pr: str,
    stage: str,
    prior_treatments: list[str],
    comorbidities: list[str],
    additional_notes: str = ""
) -> dict:
    """
    Convert structured form input into a patient dict compatible with
    the matching pipeline. Generates a natural language summary for embeddings.

    Args:
        age: Patient age in years
        diagnosis: Primary diagnosis description
        her2: HER2 status — positive/negative/unknown
        er: ER status — positive/negative/unknown
        pr: PR status — positive/negative/unknown
        stage: Disease stage — early/locally advanced/metastatic/unknown
        prior_treatments: List of prior treatments received
        comorbidities: List of relevant comorbidities
        additional_notes: Any additional clinical notes

    Returns:
        Patient dict compatible with the full matching pipeline
    """
    # Build natural language summary for embeddings
    receptor_parts = []
    if her2 != "unknown":
        receptor_parts.append(f"HER2-{her2}")
    if er != "unknown":
        receptor_parts.append(f"ER-{er}")
    if pr != "unknown":
        receptor_parts.append(f"PR-{pr}")

    receptor_str = ", ".join(receptor_parts) if receptor_parts else "unknown receptor status"
    tx_str = ", ".join(prior_treatments) if prior_treatments else "no prior treatment reported"
    comorbidity_str = ", ".join(comorbidities) if comorbidities else ""
    stage_str = f"{stage} disease" if stage != "unknown" else ""

    summary_parts = [
        f"A {age}-year-old patient with {diagnosis}." if diagnosis else f"A {age}-year-old patient.",
        f"Receptor status: {receptor_str}." if receptor_parts else "",
        f"{stage_str.capitalize()}." if stage_str else "",
        f"Prior treatments include {tx_str}." if prior_treatments else "",
        f"Comorbidities: {comorbidity_str}." if comorbidity_str else "",
        additional_notes if additional_notes else "",
    ]
    raw_summary = " ".join(p for p in summary_parts if p).strip()

    return {
        "patient_id":       "custom",
        "age":              str(age),
        "raw_summary":      raw_summary,
        "receptor_status":  {"HER2": her2, "ER": er, "PR": pr},
        "metastatic":       stage == "metastatic",
        "stage":            stage,
        "pregnant":         "pregnancy" in [c.lower() for c in comorbidities],
        "prior_treatments": prior_treatments,
        "comorbidities":    comorbidities,
    }
