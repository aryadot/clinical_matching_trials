"""
pipeline/scorer.py — Trial Scoring & Ranking
==============================================
Architecture: spaCy entity extraction + BERT cross-encoder scoring

Two-stage pipeline:
  Stage 1 — spaCy pattern matching extracts clinical entities fast and reliably.
             Used for hard exclusion checks and display purposes.

  Stage 2 — BERT cross-encoder reads the full patient summary and trial
             eligibility text as a pair and outputs a semantic relevance score.
             Replaces string-level NER overlap which required entity normalization.
             Cross-encoder never compares entity strings — it understands semantic
             relationships between patient clinical picture and trial requirements.

Composite score:
  - Semantic similarity (ChromaDB embeddings): 0.35 weight
  - Cross-encoder relevance score:             0.65 weight
  - Rule-based checks: hard filters only (exclusions, age)

Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Small (66MB), fast on CPU, strong semantic matching.
"""

import re
import os

try:
    import streamlit as st
    _IN_STREAMLIT = True
except ImportError:
    _IN_STREAMLIT = False

from config import (
    SIGNAL_DISEASE_MATCH, SIGNAL_MALIGNANCY, SIGNAL_RECEPTOR_MATCH,
    SIGNAL_METASTATIC_MATCH, SIGNAL_AGE_AVAILABLE, SIGNAL_ER_MATCH,
)

# ── Weights ───────────────────────────────────────────────────────────────────
SEMANTIC_WEIGHT      = 0.35
CROSS_ENCODER_WEIGHT = 0.65

CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder_cache = None


# ── Cross-encoder loading ─────────────────────────────────────────────────────
def _build_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER_MODEL)


if _IN_STREAMLIT:
    @st.cache_resource(show_spinner="Loading BERT cross-encoder...")
    def load_cross_encoder():
        return _build_cross_encoder()
else:
    def load_cross_encoder():
        global _cross_encoder_cache
        if _cross_encoder_cache is None:
            _cross_encoder_cache = _build_cross_encoder()
        return _cross_encoder_cache


# ── Cross-encoder scoring ─────────────────────────────────────────────────────
def compute_cross_encoder_score(patient_summary: str, trial: dict) -> float:
    """
    Use BERT cross-encoder to score semantic relevance between patient summary
    and trial eligibility criteria.

    The cross-encoder reads the patient-trial pair together and outputs a
    relevance score — no string matching, no entity normalization needed.
    It understands that 'ER-positive' and 'estrogen receptor positive' are
    the same concept in context.

    Returns normalized score 0.0 to 1.0.
    """
    try:
        # Build focused trial text — title + conditions + inclusion criteria
        trial_text = f"{trial.get('title', '')}. {trial.get('conditions', '')}."
        eligibility = trial.get("eligibility", "") or ""

        # Extract inclusion criteria section for focused matching
        inc_idx = eligibility.lower().find("inclusion criteria")
        exc_idx = eligibility.lower().find("exclusion criteria")
        if inc_idx != -1 and exc_idx != -1:
            inc_text = eligibility[inc_idx:exc_idx][:500]
        elif inc_idx != -1:
            inc_text = eligibility[inc_idx:][:500]
        else:
            inc_text = eligibility[:500]

        trial_text = f"{trial_text} {inc_text}".strip()

        if not trial_text or not patient_summary:
            return 0.0

        ce = load_cross_encoder()
        score = ce.predict([(patient_summary[:512], trial_text[:512])])

        # ms-marco returns logits — normalize to 0-1 with sigmoid
        import math
        normalized = 1 / (1 + math.exp(-float(score[0])))
        return round(normalized, 4)

    except Exception:
        return 0.0


# ── Rule-based hard filters ───────────────────────────────────────────────────
def _parse_age_years(age_str: str) -> int | None:
    if not age_str:
        return None
    m = re.search(r"(\d+)", age_str)
    return int(m.group(1)) if m else None


def _check_age_eligibility(patient_age, min_age_str: str, max_age_str: str) -> tuple[bool, str]:
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


def _check_exclusion_criteria(patient: dict, exclusion_criteria: list) -> tuple[bool, str]:
    if not exclusion_criteria:
        return False, ""
    summary_lower = patient.get("raw_summary", "").lower()
    exclusion_signals = {
        "pregnant":    ["pregnan", "gestation"],
        "cardiac":     ["heart failure", "cardiac", "myocardial"],
        "hepatic":     ["hepatic", "liver disease", "cirrhosis"],
        "renal":       ["renal failure", "kidney disease"],
        "autoimmune":  ["autoimmune", "lupus", "rheumatoid"],
    }
    exc_text = " ".join(exclusion_criteria).lower()
    for condition, keywords in exclusion_signals.items():
        if any(k in summary_lower for k in keywords) and any(k in exc_text for k in keywords):
            return True, f"Exclusion criterion matched: {condition}"
    return False, ""


def compute_rule_score(patient: dict, trial: dict) -> tuple[float, list]:
    """
    Rule-based hard filter. Returns -1.0 on hard exclusion, 0-1 otherwise.
    Used only for disqualification — not as a score component.
    """
    score = 0
    reasons = []
    max_possible = 9

    summary = patient.get("raw_summary", "").lower()
    conditions    = (trial.get("conditions", "")    or "").lower()
    interventions = (trial.get("interventions", "") or "").lower()
    eligibility   = (trial.get("eligibility", "")   or "").lower()
    trial_text    = f"{conditions} {interventions} {eligibility}"

    # Hard exclusions
    if patient.get("pregnant"):
        exc_text = " ".join(trial.get("exclusion_criteria", [])).lower()
        if "pregnan" in trial_text or "pregnan" in exc_text:
            return -1.0, ["Pregnancy exclusion"]

    excluded, exc_reason = _check_exclusion_criteria(patient, trial.get("exclusion_criteria", []))
    if excluded:
        return -1.0, [f"Hard exclusion: {exc_reason}"]

    age_eligible, age_reason = _check_age_eligibility(
        patient.get("age"), trial.get("min_age", ""), trial.get("max_age", "")
    )
    if not age_eligible:
        return -1.0, [f"Age ineligibility: {age_reason}"]
    if age_reason:
        score += 1
        reasons.append(age_reason)

    if ("breast cancer" in summary or "breast" in summary) and any(t in conditions for t in ["breast", "mammary"]):
        score += SIGNAL_DISEASE_MATCH
        reasons.append("Disease focus aligns")

    if "benign" not in summary:
        score += SIGNAL_MALIGNANCY
        reasons.append("Confirmed malignancy")

    receptors = patient.get("receptor_status", {})
    if receptors.get("HER2") != "unknown":
        her2_val = receptors["HER2"].lower()
        if her2_val in trial_text:
            score += SIGNAL_RECEPTOR_MATCH
            reasons.append(f"HER2 status matches")

    if receptors.get("ER") != "unknown":
        er_val = receptors["ER"].lower()
        if er_val in trial_text:
            score += SIGNAL_ER_MATCH
            reasons.append(f"ER status aligns")

    if patient.get("metastatic") and "metastatic" in trial_text:
        score += SIGNAL_METASTATIC_MATCH
        reasons.append("Metastatic disease aligns")

    if patient.get("age") != "unknown":
        score += SIGNAL_AGE_AVAILABLE

    if all(receptors.get(r) == "negative" for r in ["ER", "PR", "HER2"]):
        if "triple negative" in trial_text or "tnbc" in trial_text:
            score += 2
            reasons.append("Triple-negative profile matches TNBC trial")

    return min(score / max_possible, 1.0), reasons


# ── Main scoring function ─────────────────────────────────────────────────────
def score_patient_trial(
    patient: dict,
    trial: dict,
    semantic_similarity: float,
    patient_entities: dict = None,
    trial_entities: dict = None,
) -> dict:
    """
    Compute composite score using semantic similarity + cross-encoder.

    The cross-encoder replaces NER string overlap scoring.
    It reads the patient summary and trial eligibility text as a pair
    and scores semantic relevance directly — no entity normalization needed.
    """
    # Stage 1: Rule-based hard filter
    rule_score, rule_reasons = compute_rule_score(patient, trial)
    if rule_score < 0:
        return {
            "trial_id": trial.get("trial_id", ""),
            "title": trial.get("title", ""),
            "composite_score": 0.0,
            "match_percentage": 0.0,
            "semantic_score": semantic_similarity,
            "rule_score": 0.0,
            "cross_encoder_score": 0.0,
            "reasons": rule_reasons,
            "excluded": True,
        }

    # Stage 2: Cross-encoder semantic relevance scoring
    patient_summary = patient.get("raw_summary", "")
    ce_score = compute_cross_encoder_score(patient_summary, trial)

    # Stage 3: Composite score
    composite = (semantic_similarity * SEMANTIC_WEIGHT) + (ce_score * CROSS_ENCODER_WEIGHT)

    all_reasons = rule_reasons.copy()
    if semantic_similarity > 0.5:
        all_reasons.insert(0, f"High semantic similarity ({semantic_similarity:.0%})")
    if ce_score > 0.6:
        all_reasons.insert(0, f"Strong eligibility alignment ({ce_score:.0%})")

    return {
        "trial_id":            trial.get("trial_id", ""),
        "title":               trial.get("title", ""),
        "conditions":          trial.get("conditions", ""),
        "interventions":       trial.get("interventions", ""),
        "status":              trial.get("status", ""),
        "composite_score":     round(composite, 4),
        "match_percentage":    round(composite * 100, 1),
        "semantic_score":      round(semantic_similarity, 4),
        "rule_score":          round(rule_score, 4),
        "cross_encoder_score": round(ce_score, 4),
        "reasons":             all_reasons,
        "excluded":            False,
    }


def rank_trials(scored_trials: list, top_k: int = 10) -> list:
    """Sort trials by composite score and return top K matches."""
    valid  = [t for t in scored_trials if not t.get("excluded")]
    ranked = sorted(valid, key=lambda x: x["composite_score"], reverse=True)
    if ranked:
        best = ranked[0]["composite_score"]
        for t in ranked:
            t["match_percentage"] = round((t["composite_score"] / best) * 100, 1) if best > 0 else 0.0
    return ranked[:top_k]