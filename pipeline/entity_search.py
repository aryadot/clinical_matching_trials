"""
pipeline/entity_search.py — Keyword / Entity Overlap Search (List 2)
======================================================================
NOT spaCy. This is curated-term substring matching against a fixed
clinical vocabulary (biomarkers, subtypes, disease stage, treatment
classes), producing a set of matched terms per patient/trial text.

Those term sets are then ranked against each other using the
Sørensen-Dice Index, giving every trial a real rank position —
not just a present/absent flag — so this can serve as an independent
retrieval strategy (List 2) for two-way RRF fusion against the
semantic search strategy (List 1).

    Dice(X, Y) = 2 * |X ∩ Y| / (|X| + |Y|)
"""

# Curated clinical vocabulary for breast-cancer trial matching.
# Deliberately flat and explicit — no model, no black box.
CLINICAL_TERMS = [
    # Receptor / biomarker status
    "er-positive", "er-negative", "pr-positive", "pr-negative",
    "her2-positive", "her2-negative", "her2-low", "her2+",
    "triple-negative", "triple negative", "tnbc",
    "brca1", "brca2", "pik3ca", "pd-l1", "ntrk",

    # Disease characteristics
    "metastatic", "stage i", "stage ii", "stage iii", "stage iv",
    "invasive", "in situ", "recurrent", "refractory", "relapsed",
    "locally advanced", "early stage", "advanced breast cancer",

    # Prior treatment
    "chemotherapy", "neoadjuvant", "adjuvant", "radiotherapy",
    "endocrine therapy", "immunotherapy", "prior treatment",
    "trastuzumab", "pertuzumab", "tamoxifen", "paclitaxel",

    # Exclusion-relevant conditions
    "pregnant", "pregnancy", "breastfeeding", "cardiac", "hepatic",
    "renal", "autoimmune", "hiv", "active infection",
]


def extract_entities(text: str) -> set[str]:
    """
    Return the set of curated clinical terms found in `text`.
    Pure substring matching — explicitly not a trained NER model.
    """
    if not text:
        return set()
    lower = text.lower()
    return {term for term in CLINICAL_TERMS if term in lower}


def build_trial_entity_index(trials: list[dict]) -> dict[str, set[str]]:
    """
    Precompute entity sets for every trial once, so ranking all 1,046
    trials against a patient is cheap (pure set math, no re-scanning text).
    """
    index = {}
    for trial in trials:
        text = " ".join([
            trial.get("title", "") or "",
            trial.get("conditions", "") or "",
            trial.get("interventions", "") or "",
            " ".join(trial.get("inclusion_criteria", []) or []),
        ])
        index[trial.get("trial_id", "")] = extract_entities(text)
    return index


def dice_score(patient_entities: set[str], trial_entities: set[str]) -> float:
    """Sørensen-Dice Index between two entity sets. 0.0 if both are empty."""
    total = len(patient_entities) + len(trial_entities)
    if total == 0:
        return 0.0
    overlap = len(patient_entities & trial_entities)
    return round((2.0 * overlap) / total, 4)


def rank_trials_by_dice(
    patient_entities: set[str],
    trial_entity_index: dict[str, set[str]],
) -> list[tuple[str, float]]:
    """
    Rank all trials by Dice overlap with the patient's entity set.
    Returns [(trial_id, dice_score), ...] sorted highest-first.

    If patient_entities is empty, every trial scores 0.0 — this is a
    real degenerate case (no clinical terms matched in the patient
    summary), not a bug, but callers should treat an empty patient
    entity set as "List 2 has no signal" rather than a genuine ranking.
    """
    scored = [
        (trial_id, dice_score(patient_entities, trial_ents))
        for trial_id, trial_ents in trial_entity_index.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
