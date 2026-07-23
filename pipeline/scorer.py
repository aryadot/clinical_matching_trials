"""
pipeline/scorer.py — Trial Scoring Building Blocks
==============================================
Provides the two independently-usable scoring functions that
pipeline/matcher.py combines via Reciprocal Rank Fusion (no weights):

  compute_rule_score()          — deterministic hard-exclusion guardrail
                                   (pregnancy, age, exclusion criteria) plus
                                   inclusion-criteria-only biomarker phrase
                                   matching (HER2/ER/metastatic/TNBC).
                                   Returns -1.0 to hard-disqualify, else 0-1.

  compute_cross_encoder_score() — fine-tuned BERT cross-encoder relevance
                                   score for a patient/trial pair (0-1).

This file does NOT combine these into a final ranking — that fusion lives
entirely in pipeline/matcher.py via pipeline/rrf.py.

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
    SIGNAL_METASTATIC_MATCH, SIGNAL_ER_MATCH,
)

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
        # Build focused trial text — title + conditions + inclusion criteria.
        # Uses the pre-parsed inclusion_criteria field (same source of truth
        # as compute_rule_score) instead of string-searching the raw
        # eligibility blob for "inclusion criteria"/"exclusion criteria"
        # markers, which silently falls back to the ENTIRE blob (including
        # exclusion text) if a trial's formatting doesn't match those exact
        # marker phrases.
        trial_text = f"{trial.get('title', '')}. {trial.get('conditions', '')}."
        inclusion_list = trial.get("inclusion_criteria", []) or []
        inc_text = " ".join(inclusion_list)[:500]
        if not inc_text:
            # Genuine fallback: no parsed inclusion list at all for this
            # trial. Use the eligibility blob capped before any exclusion
            # marker, rather than assuming markers exist.
            eligibility = trial.get("eligibility", "") or ""
            exc_idx = eligibility.lower().find("exclusion criteria")
            inc_text = (eligibility[:exc_idx] if exc_idx != -1 else eligibility)[:500]

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


def _receptor_phrase_matches(biomarker: str, status: str, inclusion_text: str) -> bool:
    """
    Require the biomarker name AND its status to appear together as a
    specific phrase (e.g. "her2 positive", "her2+", "estrogen receptor
    negative") — never a bare "positive"/"negative" substring, which hits
    over half the entire trial database regardless of actual fit (577/1046
    trials contain "positive" somewhere; 683/1046 contain "negative").

    Checked ONLY against inclusion_criteria text, never the combined
    inclusion+exclusion blob — so a trial that EXCLUDES this status can
    never be mistaken for a match just because the status word appears in
    its exclusion language.

    The bare "her2-" / "er-" shorthand (valid clinical notation for
    negative status, e.g. "ER+ and HER2-") is matched via regex with a
    negative lookahead so it does NOT match inside compound words like
    "her2-positive", "her2-low", or "her2-directed" — verified against the
    real trial data: the naive substring version falsely matched 231 of
    1046 trials that were actually HER2-POSITIVE, not negative.
    """
    status = status.lower()
    biomarker = biomarker.lower()

    patterns = [f"{biomarker} {status}", f"{biomarker}-{status}"]
    if biomarker == "er":
        patterns += [f"estrogen receptor {status}", f"estrogen receptor-{status}"]

    if any(p in inclusion_text for p in patterns):
        return True

    if status == "positive":
        return f"{biomarker}+" in inclusion_text
    if status == "negative":
        # "her2-" / "er-" only counts when NOT followed by another letter,
        # so it can't match inside "her2-positive", "her2-low", etc.
        return re.search(rf"\b{biomarker}-(?![a-z])", inclusion_text) is not None

    return False


def compute_rule_score(patient: dict, trial: dict) -> tuple[float, list]:
    """
    Rule-based hard filter. Returns -1.0 on hard exclusion, 0-1 otherwise.
    Used only for disqualification — not as a score component.
    """
    score = 0
    reasons = []
    # max_possible = 8: age_reason(1) + disease(2) + malignancy(2) + HER2(2)
    # + ER(1) + metastatic(1) + TNBC(2) = would be 11 if all fired, but disease
    # match/malignancy/age are near-universal on a breast-cancer-only dataset —
    # see README "Known Limitations". Duplicate age-available signal removed
    # here (was double-counting the same fact as age_reason below).
    max_possible = 8

    summary = patient.get("raw_summary", "").lower()
    conditions    = (trial.get("conditions", "")    or "").lower()
    interventions = (trial.get("interventions", "") or "").lower()
    eligibility   = (trial.get("eligibility", "")   or "").lower()
    trial_text    = f"{conditions} {interventions} {eligibility}"
    # Inclusion-only text for biomarker/status matching — deliberately
    # excludes exclusion_criteria, so a trial that EXCLUDES a status can't
    # be mistaken for a match on that status. See _receptor_phrase_matches.
    inclusion_text = " ".join(trial.get("inclusion_criteria", []) or []).lower()

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
        if _receptor_phrase_matches("her2", receptors["HER2"], inclusion_text):
            score += SIGNAL_RECEPTOR_MATCH
            reasons.append("HER2 status matches")

    if receptors.get("ER") != "unknown":
        if _receptor_phrase_matches("er", receptors["ER"], inclusion_text):
            score += SIGNAL_ER_MATCH
            reasons.append("ER status aligns")

    if patient.get("metastatic") and "metastatic" in inclusion_text:
        score += SIGNAL_METASTATIC_MATCH
        reasons.append("Metastatic disease aligns")

    if all(receptors.get(r) == "negative" for r in ["ER", "PR", "HER2"]):
        if "triple negative" in inclusion_text or "tnbc" in inclusion_text:
            score += 2
            reasons.append("Triple-negative profile matches TNBC trial")

    return min(score / max_possible, 1.0), reasons


# ── Note ──────────────────────────────────────────────────────────────────────
# The old weighted-composite functions (score_patient_trial / rank_trials,
# 0.35 semantic + 0.65 cross-encoder) were removed here. They were only ever
# called by evaluate_llm_judge.py, which has also been removed — that script
# compared this pipeline against an independent LLM ranking and found weak
# agreement (Spearman ~0.22, Agreement@1 = 0%), which we concluded wasn't a
# reliable evaluation method since neither side was validated against real
# ground truth. The live matching path is entirely in pipeline/matcher.py,
# which combines compute_rule_score() and compute_cross_encoder_score()
# below via Reciprocal Rank Fusion (pipeline/rrf.py) — no hand-picked
# weights anywhere in the current live pipeline.