"""
pipeline/matcher.py — Full Weight-Free Matching Pipeline
============================================================
    [Patient Record]
         |
         +--> Strategy A: Vector DB Search        --> List 1 (semantic rank)
         +--> Strategy B: Keyword/Entity Overlap   --> List 2 (Dice rank)
         |         (Sørensen-Dice, see entity_search.py)
         v
    1. TWO-WAY RRF  --------------------------------> Top-50 Master List
         v
    2. Cross-Encoder Stage  (reads Master List pairs) --> CE-ranked list
         |
         +--> Rule-based hard-exclusion guardrail applied FIRST
         |    (pregnancy / age / exclusion-criteria keyword match —
         |     this is the existing deterministic filter, NOT the
         |     cross-encoder — see scorer.compute_rule_score)
         v
    3. DUAL-AXIS FINAL RRF  (Master-list rank x CE rank, survivors only)
         v
    Eligibility % (rescaled relative to top trial) shown to user

No stage in this pipeline uses a hand-picked weight. Every fusion step
is rank-based (RRF). The only non-rank component is the rule-based
guardrail, which is a binary gate, not a weighted score.
"""

from pipeline.embeddings import semantic_search
from pipeline.entity_search import extract_entities, rank_trials_by_dice
from pipeline.scorer import compute_rule_score, compute_cross_encoder_score
from pipeline.rrf import reciprocal_rank_fusion, fused_ranking, normalize_to_percentage

MASTER_LIST_SIZE = 50


def match_patient_to_trials(
    patient: dict,
    trials: list[dict],
    collection,
    trial_entity_index: dict[str, set],
    top_k: int = 10,
    on_progress=None,
) -> list[dict]:
    """
    Run the full weight-free pipeline for one patient and return the
    top_k trials, each annotated with every component score so the UI
    can show exactly what contributed to the final rank — nothing here
    is hidden inside an opaque composite number.
    """
    trial_lookup = {t.get("trial_id", ""): t for t in trials}

    def _progress(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    # ── Strategy A: Vector DB semantic search -> List 1 ──────────────────
    _progress(10, "Strategy A: semantic vector search...")
    patient_text = patient.get("raw_summary", "")
    semantic_hits = semantic_search(patient_text, collection, top_k=200)
    semantic_scores = {h["trial_id"]: h["similarity"] for h in semantic_hits}
    list_1 = [h["trial_id"] for h in semantic_hits]  # already best-first

    # ── Strategy B: Keyword/Entity overlap (Dice) -> List 2 ──────────────
    _progress(25, "Strategy B: keyword/entity overlap search...")
    patient_entities = extract_entities(patient_text)
    dice_ranked = rank_trials_by_dice(patient_entities, trial_entity_index)
    dice_scores = dict(dice_ranked)
    list_2 = [tid for tid, score in dice_ranked if score > 0]  # only real overlaps count as "ranked"

    # ── Stage 1: Two-way RRF -> Top-50 Master List ───────────────────────
    _progress(40, "Fusing List 1 & List 2 (two-way RRF)...")
    master_rrf_scores = reciprocal_rank_fusion(list_1, list_2)
    master_order = fused_ranking(master_rrf_scores)[:MASTER_LIST_SIZE]

    # ── Rule-based hard-exclusion guardrail (deterministic, not CE) ──────
    _progress(55, "Applying rule-based safety guardrail...")
    disqualified = set()
    rule_results = {}
    for trial_id in master_order:
        trial = trial_lookup.get(trial_id, {})
        rule_score, rule_reasons = compute_rule_score(patient, trial)
        rule_results[trial_id] = (rule_score, rule_reasons)
        if rule_score < 0:
            disqualified.add(trial_id)

    survivors = [tid for tid in master_order if tid not in disqualified]

    # ── Stage 2: Cross-encoder deep read of survivors ────────────────────
    _progress(70, "Cross-encoder deep-reading survivors...")
    ce_scores = {}
    for trial_id in survivors:
        trial = trial_lookup.get(trial_id, {})
        ce_scores[trial_id] = compute_cross_encoder_score(patient_text, trial)
    ce_order = sorted(survivors, key=lambda tid: ce_scores.get(tid, 0.0), reverse=True)

    # ── Stage 3: Dual-axis final RRF (master rank x CE rank) ─────────────
    _progress(90, "Final dual-axis fusion...")
    final_rrf_scores = reciprocal_rank_fusion(
        [tid for tid in master_order if tid in survivors],  # master rank, survivors only
        ce_order,
        exclude=disqualified,
    )
    final_pct = normalize_to_percentage(final_rrf_scores)
    final_order = fused_ranking(final_rrf_scores)

    # ── Assemble transparent results ─────────────────────────────────────
    results = []
    for trial_id in final_order[:top_k]:
        trial = trial_lookup.get(trial_id, {})
        rule_score, rule_reasons = rule_results.get(trial_id, (0.0, []))
        results.append({
            "trial_id": trial_id,
            "title": trial.get("title", ""),
            "conditions": trial.get("conditions", ""),
            "interventions": trial.get("interventions", ""),
            "status": trial.get("status", ""),
            "eligibility_pct": final_pct.get(trial_id, 0.0),
            "final_rrf_score": round(final_rrf_scores.get(trial_id, 0.0), 5),
            "semantic_score": round(semantic_scores.get(trial_id, 0.0), 4),
            "dice_score": round(dice_scores.get(trial_id, 0.0), 4),
            "cross_encoder_score": round(ce_scores.get(trial_id, 0.0), 4),
            "master_rrf_score": round(master_rrf_scores.get(trial_id, 0.0), 5),
            "reasons": rule_reasons,
            "matched_entities": sorted(patient_entities & trial_entity_index.get(trial_id, set())),
        })

    _progress(100, "Done.")
    return results
