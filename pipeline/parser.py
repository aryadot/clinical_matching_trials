"""
pipeline/parser.py — Patient profile extraction and trial text preparation
Extracts structured clinical features from raw patient summaries
and prepares trial text for embedding.
"""

import json
import re
from pathlib import Path


def extract_age(text: str) -> int | str:
    match = re.search(r"(\d{2})[- ]?year[- ]?old", text.lower())
    return int(match.group(1)) if match else "unknown"


def extract_pregnancy(text: str) -> bool:
    keywords = ["pregnan", "gestation", "trimester", "intrauterine"]
    return any(k in text.lower() for k in keywords)


def extract_receptor_status(text: str) -> dict:
    status = {"ER": "unknown", "PR": "unknown", "HER2": "unknown"}
    lower = text.lower()

    if "er-positive" in lower or "er positive" in lower:
        status["ER"] = "positive"
    elif "er-negative" in lower or "er negative" in lower:
        status["ER"] = "negative"

    if "pr-positive" in lower or "pr positive" in lower:
        status["PR"] = "positive"
    elif "pr-negative" in lower or "pr negative" in lower:
        status["PR"] = "negative"

    if "her2-positive" in lower or "her2 positive" in lower or "her2+" in lower:
        status["HER2"] = "positive"
    elif "her2-low" in lower:
        status["HER2"] = "low"
    elif "her2-negative" in lower or "her2 negative" in lower:
        status["HER2"] = "negative"

    if "triple-negative" in lower or "triple negative" in lower or "tnbc" in lower:
        status["ER"] = "negative"
        status["PR"] = "negative"
        status["HER2"] = "negative"

    return status


def extract_metastatic(text: str) -> bool:
    return "metastatic" in text.lower() or "stage iv" in text.lower()


def extract_stage(text: str) -> str:
    lower = text.lower()
    for stage in ["stage iv", "stage iii", "stage ii", "stage i"]:
        if stage in lower:
            return stage.upper()
    if "metastatic" in lower:
        return "STAGE IV"
    if "locally advanced" in lower:
        return "STAGE III"
    return "unknown"


def extract_prior_therapy(text: str) -> list[str]:
    keywords = ["chemotherapy", "endocrine therapy", "targeted therapy",
                "trastuzumab", "pertuzumab", "tamoxifen", "letrozole",
                "radiation", "surgery", "mastectomy", "lumpectomy"]
    return [k for k in keywords if k in text.lower()]


def extract_comorbidities(text: str) -> list[str]:
    keywords = ["autoimmune", "heart failure", "cardiac", "diabetes",
                "hypertension", "renal", "hepatic", "liver disease"]
    return [k for k in keywords if k in text.lower()]


def parse_patient(patient: dict) -> dict:
    """Parse a raw patient dict into structured clinical profile."""
    summary = patient["summary"]
    return {
        "patient_id": patient["patient_id"],
        "age": extract_age(summary),
        "pregnant": extract_pregnancy(summary),
        "receptor_status": extract_receptor_status(summary),
        "metastatic": extract_metastatic(summary),
        "stage": extract_stage(summary),
        "prior_therapy": extract_prior_therapy(summary),
        "comorbidities": extract_comorbidities(summary),
        "raw_summary": summary,
    }


def build_patient_embedding_text(patient: dict) -> str:
    """Build a text representation of a patient profile optimized for embedding similarity."""
    parts = []
    parts.append(f"Patient with breast cancer.")

    if patient["age"] != "unknown":
        parts.append(f"Age: {patient['age']} years old.")

    rs = patient["receptor_status"]
    receptor_parts = []
    for r, v in rs.items():
        if v != "unknown":
            receptor_parts.append(f"{r}-{v}")
    if receptor_parts:
        parts.append(f"Receptor status: {', '.join(receptor_parts)}.")

    if patient["metastatic"]:
        parts.append("Metastatic disease.")

    if patient["stage"] != "unknown":
        parts.append(f"Disease stage: {patient['stage']}.")

    if patient["prior_therapy"]:
        parts.append(f"Prior therapy: {', '.join(patient['prior_therapy'])}.")

    if patient["comorbidities"]:
        parts.append(f"Comorbidities: {', '.join(patient['comorbidities'])}.")

    if patient["pregnant"]:
        parts.append("Currently pregnant.")

    return " ".join(parts)


def build_trial_embedding_text(trial: dict) -> str:
    """Build a text representation of a trial optimized for embedding similarity."""
    parts = []
    parts.append(f"Clinical trial: {trial.get('title', '')}.")
    parts.append(f"Conditions: {trial.get('conditions', '')}.")
    parts.append(f"Interventions: {trial.get('interventions', '')}.")

    inc = trial.get("inclusion_criteria", [])
    if inc:
        parts.append(f"Inclusion criteria: {' '.join(inc[:5])}.")

    exc = trial.get("exclusion_criteria", [])
    if exc:
        parts.append(f"Exclusion criteria: {' '.join(exc[:3])}.")

    return " ".join(parts)


def load_patients(path: Path) -> list[dict]:
    """Load and parse raw patient summaries."""
    with open(path) as f:
        raw = json.load(f)
    return [parse_patient(p) for p in raw]


def load_trials(path: Path) -> list[dict]:
    """Load parsed trials."""
    with open(path) as f:
        return json.load(f)
