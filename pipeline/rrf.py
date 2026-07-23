"""
pipeline/rrf.py — Reciprocal Rank Fusion
===========================================
Combines two or more ranked lists of the same items WITHOUT hand-picked
weights. Each list only ever contributes its RANK position, never its
raw score — so lists on completely different scales (cosine similarity,
Dice overlap, cross-encoder logits) can be fused fairly without needing
to decide "how much do I trust list A vs list B."

    RRF(item) = sum over lists of  1 / (k + rank_in_that_list)

k=60 is the standard constant from the original RRF paper (Cormack et
al., 2009) — a smoothing constant, not something tuned per-project.
"""

RRF_K = 60


def reciprocal_rank_fusion(
    *ranked_id_lists: list[str],
    k: int = RRF_K,
    exclude: set[str] | None = None,
) -> dict[str, float]:
    """
    ranked_id_lists: one or more lists of trial_ids, each already sorted
                      best-first by that list's own scoring method.
    exclude:          trial_ids to drop before fusion (e.g. trials that
                      failed the rule-based hard-exclusion gate).

    Returns {trial_id: rrf_score}, NOT sorted — sort by value to rank.
    """
    exclude = exclude or set()
    fused: dict[str, float] = {}

    for ranked_ids in ranked_id_lists:
        rank = 0
        for trial_id in ranked_ids:
            if trial_id in exclude:
                continue
            rank += 1  # rank position among survivors only, no gaps
            fused[trial_id] = fused.get(trial_id, 0.0) + (1.0 / (k + rank))

    return fused


def fused_ranking(fused_scores: dict[str, float]) -> list[str]:
    """Return trial_ids sorted best-first by their fused RRF score."""
    return [tid for tid, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]


def normalize_to_percentage(scores: dict[str, float]) -> dict[str, float]:
    """
    Rescale relative to the top score in the set, matching how the rest
    of the app displays match strength (top trial = 100%, others relative
    to it — NOT an absolute eligibility probability).
    """
    if not scores:
        return {}
    best = max(scores.values())
    if best <= 0:
        return {tid: 0.0 for tid in scores}
    return {tid: round((score / best) * 100, 1) for tid, score in scores.items()}
