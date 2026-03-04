"""
pipeline/ner.py — Clinical Named Entity Recognition
Extracts biomarkers, cancer types, treatments, and stages from clinical text
using spaCy NER + custom domain-specific pattern matching.
"""

import re
import streamlit as st
from config import BIOMARKERS, CANCER_TYPES, TREATMENTS, STAGES


@st.cache_resource(show_spinner="Loading NER model...")
def load_ner_model():
    """Load spaCy model for general NER."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def extract_clinical_entities(text: str) -> dict:
    """
    Extract clinical entities from text using both spaCy NER
    and custom domain-specific pattern matching.

    Returns:
        dict with keys: biomarkers, cancer_types, treatments, stages, organizations, misc_entities
    """
    lower = text.lower()

    # Domain-specific extraction (pattern matching on clinical terms)
    found_biomarkers = [b for b in BIOMARKERS if b.lower() in lower]
    found_cancers = [c for c in CANCER_TYPES if c.lower() in lower]
    found_treatments = [t for t in TREATMENTS if t.lower() in lower]
    found_stages = [s for s in STAGES if s.lower() in lower]

    # spaCy NER for organizations, people, etc.
    nlp = load_ner_model()
    doc = nlp(text[:5000])  # Limit to avoid long processing

    organizations = list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))
    misc_entities = list(set(ent.text for ent in doc.ents if ent.label_ in ("PRODUCT", "EVENT", "WORK_OF_ART")))

    return {
        "biomarkers": list(set(found_biomarkers)),
        "cancer_types": list(set(found_cancers)),
        "treatments": list(set(found_treatments)),
        "stages": list(set(found_stages)),
        "organizations": organizations[:5],
        "misc_entities": misc_entities[:5],
    }


def compute_entity_overlap(patient_entities: dict, trial_entities: dict) -> float:
    """
    Compute an overlap score between patient and trial entities.
    Returns a score from 0.0 to 1.0 based on how many clinical entities match.
    """
    score = 0.0
    max_score = 0.0

    # Biomarker overlap (highest weight — most clinically relevant)
    p_bio = set(b.lower() for b in patient_entities.get("biomarkers", []))
    t_bio = set(b.lower() for b in trial_entities.get("biomarkers", []))
    if p_bio or t_bio:
        max_score += 3.0
        overlap = len(p_bio & t_bio)
        if overlap > 0:
            score += 3.0 * (overlap / max(len(p_bio), len(t_bio), 1))

    # Cancer type overlap
    p_cancer = set(c.lower() for c in patient_entities.get("cancer_types", []))
    t_cancer = set(c.lower() for c in trial_entities.get("cancer_types", []))
    if p_cancer or t_cancer:
        max_score += 2.0
        overlap = len(p_cancer & t_cancer)
        if overlap > 0:
            score += 2.0 * (overlap / max(len(p_cancer), len(t_cancer), 1))

    # Treatment overlap
    p_treat = set(t.lower() for t in patient_entities.get("treatments", []))
    t_treat = set(t.lower() for t in trial_entities.get("treatments", []))
    if p_treat or t_treat:
        max_score += 1.0
        overlap = len(p_treat & t_treat)
        if overlap > 0:
            score += 1.0 * (overlap / max(len(p_treat), len(t_treat), 1))

    # Stage overlap
    p_stage = set(s.lower() for s in patient_entities.get("stages", []))
    t_stage = set(s.lower() for s in trial_entities.get("stages", []))
    if p_stage or t_stage:
        max_score += 2.0
        overlap = len(p_stage & t_stage)
        if overlap > 0:
            score += 2.0 * (overlap / max(len(p_stage), len(t_stage), 1))

    return score / max_score if max_score > 0 else 0.0
