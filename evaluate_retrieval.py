"""
evaluate_retrieval.py — Retrieval Quality Evaluation for Clinical Trial Navigator

Metrics computed:
  - Precision@k  : fraction of top-k retrieved trials that are relevant
  - Recall@k     : fraction of all relevant trials captured in top-k
  - MRR          : Mean Reciprocal Rank (how high the first relevant result ranks)
  - NDCG@k       : Normalized Discounted Cumulative Gain (quality of ranking)

Ground truth is built using rule-based + NER scoring only (no semantic component)
to avoid bias from a fixed semantic similarity value.
"""

import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.embeddings import get_chroma_client, get_collection, index_trials, semantic_search
from pipeline.ner import extract_clinical_entities, compute_entity_overlap
from pipeline.scorer import score_patient_trial, rank_trials, compute_rule_score
from pipeline.parser import parse_patient

K_VALUES = [1, 3, 5, 10]
RELEVANCE_THRESHOLD = 0.50
TOP_RETRIEVAL = 50


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
    print("Building ground truth via rule + NER scoring...")
    ground_truth = {}
    for raw_p in patients:
        patient = parse_patient(raw_p)
        patient_entities = extract_clinical_entities(patient.get("raw_summary", ""))
        relevant = set()
        for trial in trials:
            trial_text = f"{trial.get('conditions', '')} {trial.get('interventions', '')} {trial.get('eligibility', '')}"
            trial_entities = extract_clinical_entities(trial_text)
            rule_score, _ = compute_rule_score(patient, trial)
            if rule_score < 0:
                continue
            ner_score = compute_entity_overlap(patient_entities, trial_entities)
            combined = (rule_score * 0.6) + (ner_score * 0.4)
            if combined >= RELEVANCE_THRESHOLD:
                tid = trial.get("trial_id") or trial.get("nct_id") or trial.get("NCTId", "")
                relevant.add(tid)
        ground_truth[raw_p["patient_id"]] = relevant
        print(f"  {raw_p['patient_id']}: {len(relevant)} relevant trials")
    return ground_truth


def retrieve_for_patient(patient, trials_lookup, collection, top_k=TOP_RETRIEVAL):
    parsed = parse_patient(patient)
    query_text = parsed.get("raw_summary", patient.get("summary", ""))
    patient_entities = extract_clinical_entities(query_text)
    candidates = semantic_search(query_text, collection, top_k=top_k)
    scored = []
    for c in candidates:
        trial = trials_lookup.get(c["trial_id"])
        if not trial:
            continue
        trial_text = f"{trial.get('conditions', '')} {trial.get('interventions', '')} {trial.get('eligibility', '')}"
        trial_entities = extract_clinical_entities(trial_text)
        result = score_patient_trial(
            patient=parsed, trial=trial,
            semantic_similarity=c["similarity"],
            patient_entities=patient_entities, trial_entities=trial_entities,
        )
        scored.append(result)
    ranked = rank_trials(scored, top_k=top_k)
    return [r["trial_id"] for r in ranked]


def precision_at_k(retrieved, relevant, k):
    if not retrieved or not relevant:
        return 0.0
    return sum(1 for t in retrieved[:k] if t in relevant) / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return sum(1 for t in retrieved[:k] if t in relevant) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for i, t in enumerate(retrieved):
        if t in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, t in enumerate(top_k) if t in relevant)
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

    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    mrr_scores = []

    print("\nRunning retrieval evaluation...")
    for raw_p in raw_patients:
        pid = raw_p["patient_id"]
        relevant = ground_truth.get(pid, set())
        if not relevant:
            print(f"  {pid}: no relevant trials, skipping")
            continue
        retrieved = retrieve_for_patient(raw_p, trials_lookup, collection)
        rr = reciprocal_rank(retrieved, relevant)
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