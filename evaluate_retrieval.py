"""
evaluate_retrieval.py — Retrieval Quality Evaluation for Clinical Trial Navigator

Architecture being evaluated (matches pipeline/matcher.py, the live pipeline):
  Strategy A: ChromaDB semantic search               -> List 1
  Strategy B: Keyword/Entity overlap (Sørensen-Dice)  -> List 2
  Stage 1:    Two-way Reciprocal Rank Fusion          -> Top-50 master list
  Stage 2:    Rule-based hard-exclusion guardrail (age, exclusion criteria, pregnancy)
  Stage 3:    Cross-encoder BERT scores survivors
  Stage 4:    Dual-axis final RRF (master rank x cross-encoder rank)

No weighted composite anywhere in this evaluation — matches the live app exactly.

Metrics:
  MRR        — Mean Reciprocal Rank
  Precision@k — Fraction of top-k retrieved that are relevant
  Recall@k   — Fraction of all relevant trials captured in top-k
  NDCG@k     — Normalized Discounted Cumulative Gain

NOTE ON GROUND TRUTH: "relevant" trials below are still defined by the
same rule-based compute_rule_score() threshold used elsewhere in the
pipeline — this evaluation checks internal consistency of the new
architecture, not clinical correctness. See README "Known Limitations."
"""

import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.embeddings import get_chroma_client, get_collection, index_trials
from pipeline.entity_search import build_trial_entity_index
from pipeline.matcher import match_patient_to_trials
from pipeline.scorer import compute_rule_score
from pipeline.parser import parse_patient

K_VALUES           = [1, 3, 5, 10]
# Generic signals alone (disease match + malignancy + age-known) cap at
# 5/8 = 0.625 on this breast-cancer-only dataset — see pipeline/scorer.py.
# Threshold set above that ceiling so a trial can ONLY count as "relevant"
# if at least one real patient-specific signal (HER2/ER/metastatic/TNBC
# match) also fires, not generic topical relevance alone.
RELEVANCE_THRESHOLD = 0.7
TOP_RETRIEVAL      = 50


def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "data/patients/synthetic_patients.json")) as f:
        raw_patients = json.load(f)
    trials_path = os.path.join(base, "data/trials/parsed_trials.json")
    if os.path.exists(trials_path):
        with open(trials_path) as f:
            trials = json.load(f)
    else:
        import pandas as pd
        df = pd.read_csv(os.path.join(base, "data/trials/breast_cancer_trials.csv"))
        trials = df.to_dict(orient="records")
    return raw_patients, trials


def build_ground_truth(patients, trials):
    """
    Build ground truth using rule scoring only — no semantic component
    to avoid bias from a fixed similarity value.
    """
    print("\nBuilding ground truth via rule scoring...")
    ground_truth = {}
    for raw_p in patients:
        patient = parse_patient(raw_p)
        relevant = set()
        for trial in trials:
            tid = trial.get("trial_id", "")
            rule_score, _ = compute_rule_score(patient, trial)
            if rule_score >= RELEVANCE_THRESHOLD:
                relevant.add(tid)
        ground_truth[raw_p["patient_id"]] = relevant
        print(f"  {raw_p['patient_id']}: {len(relevant)} relevant trials")
    return ground_truth


def retrieve_for_patient(patient, trials_lookup, collection, trial_entity_index, top_k=TOP_RETRIEVAL):
    """Run the actual live pipeline (matcher.match_patient_to_trials) for evaluation."""
    trials = list(trials_lookup.values())
    results = match_patient_to_trials(
        patient=parse_patient(patient),
        trials=trials,
        collection=collection,
        trial_entity_index=trial_entity_index,
        top_k=top_k,
    )
    return [r["trial_id"] for r in results]


def precision_at_k(retrieved, relevant, k):
    return sum(1 for t in retrieved[:k] if t in relevant) / k if retrieved and relevant else 0.0

def recall_at_k(retrieved, relevant, k):
    return sum(1 for t in retrieved[:k] if t in relevant) / len(relevant) if relevant else 0.0

def reciprocal_rank(retrieved, relevant):
    for i, t in enumerate(retrieved):
        if t in relevant:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    dcg  = sum(1.0 / math.log2(i + 2) for i, t in enumerate(top_k) if t in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate():
    raw_patients, trials = load_data()

    trials_lookup = {}
    for t in trials:
        tid = t.get("trial_id") or t.get("nct_id") or t.get("NCTId", "")
        t["trial_id"] = tid
        trials_lookup[tid] = t

    ground_truth = build_ground_truth(raw_patients, trials)

    print("\nIndexing trials into ChromaDB...")
    trial_texts = [
        f"{t.get('conditions', '')} {t.get('interventions', '')} {t.get('eligibility', '')} {t.get('title', '')}"
        for t in trials
    ]
    client = get_chroma_client()
    index_trials(trials, trial_texts)
    collection = get_collection(client)

    print("\nBuilding keyword/entity index (Strategy B)...")
    trial_entity_index = build_trial_entity_index(trials)

    results   = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    mrr_scores = []

    print("\nRunning retrieval evaluation...")
    for raw_p in raw_patients:
        pid      = raw_p["patient_id"]
        relevant = ground_truth.get(pid, set())
        if not relevant:
            print(f"  {pid}: no relevant trials, skipping")
            continue
        retrieved = retrieve_for_patient(raw_p, trials_lookup, collection, trial_entity_index)
        rr        = reciprocal_rank(retrieved, relevant)
        mrr_scores.append(rr)
        for k in K_VALUES:
            results[k]["precision"].append(precision_at_k(retrieved, relevant, k))
            results[k]["recall"].append(recall_at_k(retrieved, relevant, k))
            results[k]["ndcg"].append(ndcg_at_k(retrieved, relevant, k))
        print(f"  {pid}: retrieved {len(retrieved)}, relevant {len(relevant)}, RR={rr:.3f}")

    if not mrr_scores:
        print("No patients had relevant trials. Try lowering RELEVANCE_THRESHOLD.")
        return

    print("\n" + "=" * 55)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 55)
    print(f"Patients evaluated : {len(mrr_scores)}")
    print(f"Relevance threshold: {RELEVANCE_THRESHOLD}")
    print(f"MRR                : {sum(mrr_scores)/len(mrr_scores):.4f}\n")
    print(f"{'k':<6} {'P@k':<10} {'R@k':<10} {'NDCG@k':<10}")
    print("-" * 36)
    for k in K_VALUES:
        p = results[k]["precision"]
        r = results[k]["recall"]
        n = results[k]["ndcg"]
        print(f"{k:<6} {sum(p)/len(p):<10.4f} {sum(r)/len(r):<10.4f} {sum(n)/len(n):<10.4f}")
    print("=" * 55)


if __name__ == "__main__":
    evaluate()