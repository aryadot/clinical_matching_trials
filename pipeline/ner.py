"""
pipeline/ner.py — Clinical Named Entity Recognition
=====================================================
Pattern-based entity extraction using domain-specific clinical term lists.

Role in the pipeline:
  - Extracts clinical entities for display and explanation purposes
  - Used in hard exclusion checks within the rule scorer
  - Entity overlap is NOT used for scoring — the cross-encoder in scorer.py
    handles semantic relevance directly without string matching

Fast, deterministic, and reliable for known breast cancer terminology.
"""

try:
    import streamlit as st
    _IN_STREAMLIT = True
except ImportError:
    _IN_STREAMLIT = False

from config import BIOMARKERS, CANCER_TYPES, TREATMENTS, STAGES


def extract_clinical_entities(text: str) -> dict:
    """
    Extract clinical entities using pattern matching against curated term lists.

    Returns:
        dict with keys: biomarkers, cancer_types, treatments, stages,
                        organizations, misc_entities
    """
    lower = text.lower()
    return {
        "biomarkers":    [b for b in BIOMARKERS   if b.lower() in lower],
        "cancer_types":  [c for c in CANCER_TYPES if c.lower() in lower],
        "treatments":    [t for t in TREATMENTS   if t.lower() in lower],
        "stages":        [s for s in STAGES       if s.lower() in lower],
        "organizations": [],
        "misc_entities": [],
    }


def compute_entity_overlap(patient_entities: dict, trial_entities: dict) -> float:
    """
    Compute weighted overlap score between patient and trial entities.
    Kept for backwards compatibility — not used in scoring pipeline.
    Cross-encoder in scorer.py handles semantic matching instead.
    Returns 0.0 to 1.0.
    """
    score = 0.0
    max_score = 0.0

    def _overlap(p_list, t_list, weight):
        nonlocal score, max_score
        p_set = set(x.lower() for x in p_list)
        t_set = set(x.lower() for x in t_list)
        if p_set or t_set:
            max_score += weight
            n = len(p_set & t_set)
            if n > 0:
                score += weight * (n / max(len(p_set), len(t_set), 1))

    _overlap(patient_entities.get("biomarkers",   []), trial_entities.get("biomarkers",   []), 3.0)
    _overlap(patient_entities.get("cancer_types", []), trial_entities.get("cancer_types", []), 2.0)
    _overlap(patient_entities.get("treatments",   []), trial_entities.get("treatments",   []), 1.0)
    _overlap(patient_entities.get("stages",       []), trial_entities.get("stages",       []), 2.0)

    return score / max_score if max_score > 0 else 0.0