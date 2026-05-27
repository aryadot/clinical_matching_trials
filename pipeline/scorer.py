"""
pipeline/scorer.py — Trial Scoring & Ranking
Combines three scoring signals:
  1. Semantic similarity (embedding cosine distance)
  2. Rule-based clinical signals (from original project)
  3. NER entity overlap (biomarkers, cancer types, stages)
"""

from config import (
    SEMANTIC_WEIGHT, RULE_WEIGHT, NER_WEIGHT,
    SIGNAL_DISEASE_MATCH, SIGNAL_MALIGNANCY, SIGNAL_RECEPTOR_MATCH,
    SIGNAL_METASTATIC_MATCH, SIGNAL_AGE_AVAILABLE, SIGNAL_ER_MATCH,
    PENALTY_PREGNANCY,
)
from pipeline.ner import extract_clinical_entities, compute_entity_overlap


import re


def _parse_age_years(age_str: str) -> int | None:
    """Parse age string like '18 Years' or '65 Years' into int."""
    if not age_str:
        return None
    m = re.search(r"(\d+)", age_str)
    return int(m.group(1)) if m else None


def _check_age_eligibility(patient_age, min_age_str: str, max_age_str: str) -> tuple[bool, str]:
    """
    Check if patient age falls within trial age bounds.
    Returns (eligible, reason).
    """
    if patient_age == "unknown":
        return True, ""
    min_age = _parse_age_years(min_age_str)
    max_age = _parse_age_years(max_age_str)
    age = int(patient_age)
    if min_age and age < min_age:
        return False, f"Patient age {age} below trial minimum {min_age}"
    if max_age and age > max_age:
        return False, f"Patient age {age} above trial maximum {max_age}"
    return True, f"Age {age} within trial range"


def _check_exclusion_criteria(patient: dict, exclusion_criteria: list[str]) -> tuple[bool, str]:
    """
    Check patient summary against trial exclusion criteria text.
    Returns (excluded, reason).
    """
    if not exclusion_criteria:
        return False, ""

    summary_lower = patient.get("raw_summary", "").lower()
    exclusion_signals = {
        "pregnant": ["pregnan", "gestation"],
        "cardiac":  ["heart failure", "cardiac", "myocardial"],
        "hepatic":  ["hepatic", "liver disease", "cirrhosis"],
        "renal":    ["renal failure", "kidney disease"],
        "autoimmune": ["autoimmune", "lupus", "rheumatoid"],
    }

    exc_text = " ".join(exclusion_criteria).lower()
    for condition, keywords in exclusion_signals.items():
        patient_has = any(k in summary_lower for k in keywords)
        trial_excludes = any(k in exc_text for k in keywords)
        if patient_has and trial_excludes:
            return True, f"Exclusion criterion matched: {condition}"

    return False, ""


def compute_rule_score(patient: dict, trial: dict) -> tuple[float, list[str]]:
    """
    Rule-based scoring combining disease signals, receptor matching,
    age eligibility, and exclusion criteria parsing.
    Returns (normalized_score, reasons).
    """
    score = 0
    reasons = []
    max_possible = 12  # Increased to account for new signals

    summary = patient["raw_summary"].lower()
    conditions   = (trial.get("conditions", "")   or "").lower()
    interventions = (trial.get("interventions", "") or "").lower()
    eligibility  = (trial.get("eligibility", "")  or "").lower()
    trial_text   = f"{conditions} {interventions} {eligibility}".lower()

    # ── Hard exclusions ────────────────────────────────────────────────────────

    # Pregnancy exclusion
    if patient.get("pregnant"):
        exc_criteria = trial.get("exclusion_criteria", [])
        exc_text = " ".join(exc_criteria).lower() if exc_criteria else ""
        if "pregnan" in trial_text or "pregnan" in exc_text:
            return -1.0, ["Pregnancy exclusion — hard disqualifier"]

    # Exclusion criteria check
    excluded, exc_reason = _check_exclusion_criteria(
        patient, trial.get("exclusion_criteria", [])
    )
    if excluded:
        return -1.0, [f"Hard exclusion: {exc_reason}"]

    # ── Age eligibility ───────────────────────────────────────────────────────
    age_eligible, age_reason = _check_age_eligibility(
        patient.get("age"),
        trial.get("min_age", ""),
        trial.get("max_age", "")
    )
    if not age_eligible:
        return -1.0, [f"Age ineligibility: {age_reason}"]
    if age_reason:
        score += 1
        reasons.append(age_reason)

    # ── Core disease relevance ────────────────────────────────────────────────
    if "breast cancer" in summary or "breast" in summary:
        if any(term in conditions for term in ["breast", "mammary"]):
            score += SIGNAL_DISEASE_MATCH
            reasons.append("Disease focus aligns (breast cancer)")

    # Malignancy check
    if "benign" not in summary:
        score += SIGNAL_MALIGNANCY
        reasons.append("Confirmed malignancy")

    # Receptor matching
    receptors = patient.get("receptor_status", {})

    if receptors.get("HER2") != "unknown":
        her2_val = receptors["HER2"].lower()
        if her2_val in trial_text or f"her2-{her2_val}" in trial_text or f"her2 {her2_val}" in trial_text:
            score += SIGNAL_RECEPTOR_MATCH
            reasons.append(f"HER2 status ({receptors['HER2']}) matches trial criteria")

    if receptors.get("ER") != "unknown":
        er_val = receptors["ER"].lower()
        if er_val in trial_text or f"er-{er_val}" in trial_text:
            score += SIGNAL_ER_MATCH
            reasons.append(f"ER status ({receptors['ER']}) aligns with trial")

    # Metastatic relevance
    if patient.get("metastatic") and "metastatic" in trial_text:
        score += SIGNAL_METASTATIC_MATCH
        reasons.append("Metastatic disease aligns with trial population")

    # Age signal
    if patient.get("age") != "unknown":
        score += SIGNAL_AGE_AVAILABLE
        reasons.append("Age information available for eligibility check")

    # Triple-negative specific matching
    if all(receptors.get(r) == "negative" for r in ["ER", "PR", "HER2"]):
        if "triple negative" in trial_text or "tnbc" in trial_text:
            score += 2
            reasons.append("Triple-negative profile matches TNBC trial")

    normalized = score / max_possible if max_possible > 0 else 0.0
    return min(normalized, 1.0), reasons


def score_patient_trial(
    patient: dict,
    trial: dict,
    semantic_similarity: float,
    patient_entities: dict = None,
    trial_entities: dict = None,
) -> dict:
    """
    Compute composite score combining all three signals.

    Args:
        patient: Parsed patient profile
        trial: Trial dict with conditions, interventions, etc.
        semantic_similarity: Cosine similarity from embedding search (0-1)
        patient_entities: Pre-extracted clinical entities from patient text
        trial_entities: Pre-extracted clinical entities from trial text

    Returns:
        dict with composite_score, match_percentage, component scores, and reasons
    """
    # 1. Rule-based score
    rule_score, rule_reasons = compute_rule_score(patient, trial)

    # Hard exclusion
    if rule_score < 0:
        return {
            "trial_id": trial.get("trial_id", ""),
            "title": trial.get("title", ""),
            "composite_score": 0.0,
            "match_percentage": 0.0,
            "semantic_score": semantic_similarity,
            "rule_score": 0.0,
            "ner_score": 0.0,
            "reasons": rule_reasons,
            "excluded": True,
        }

    # 2. NER entity overlap score
    ner_score = 0.0
    ner_reasons = []
    if patient_entities and trial_entities:
        ner_score = compute_entity_overlap(patient_entities, trial_entities)
        if ner_score > 0.3:
            # Find which entities overlapped
            p_bio = set(b.lower() for b in patient_entities.get("biomarkers", []))
            t_bio = set(b.lower() for b in trial_entities.get("biomarkers", []))
            shared_bio = p_bio & t_bio
            if shared_bio:
                ner_reasons.append(f"Shared biomarkers: {', '.join(shared_bio)}")

            p_treat = set(t.lower() for t in patient_entities.get("treatments", []))
            t_treat = set(t.lower() for t in trial_entities.get("treatments", []))
            shared_treat = p_treat & t_treat
            if shared_treat:
                ner_reasons.append(f"Treatment overlap: {', '.join(shared_treat)}")

    # 3. Composite score
    composite = (
        (semantic_similarity * SEMANTIC_WEIGHT) +
        (rule_score * RULE_WEIGHT) +
        (ner_score * NER_WEIGHT)
    )

    # Combine all reasons
    all_reasons = rule_reasons + ner_reasons
    if semantic_similarity > 0.5:
        all_reasons.insert(0, f"High semantic similarity ({semantic_similarity:.0%}) to patient profile")

    return {
        "trial_id": trial.get("trial_id", ""),
        "title": trial.get("title", ""),
        "conditions": trial.get("conditions", ""),
        "interventions": trial.get("interventions", ""),
        "status": trial.get("status", ""),
        "composite_score": round(composite, 4),
        "match_percentage": round(composite * 100, 1),
        "semantic_score": round(semantic_similarity, 4),
        "rule_score": round(rule_score, 4),
        "ner_score": round(ner_score, 4),
        "reasons": all_reasons,
        "excluded": False,
    }


def rank_trials(scored_trials: list[dict], top_k: int = 10) -> list[dict]:
    """Sort trials by composite score and return top K matches."""
    valid = [t for t in scored_trials if not t.get("excluded")]
    ranked = sorted(valid, key=lambda x: x["composite_score"], reverse=True)

    # Normalize match percentages relative to best match
    if ranked:
        best = ranked[0]["composite_score"]
        for t in ranked:
            t["match_percentage"] = round((t["composite_score"] / best) * 100, 1) if best > 0 else 0.0

    return ranked[:top_k]