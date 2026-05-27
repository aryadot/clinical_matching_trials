"""
evaluate_llm_judge.py — LLM-as-Judge Evaluation for Clinical Trial Navigator

Uses Llama 4 Scout via Groq as an independent external evaluator to assess:
  - Clinical Relevance  : How clinically appropriate is this trial for the patient? (1-5)
  - Eligibility Alignment: How well does the patient meet trial eligibility criteria? (1-5)

This complements existing retrieval metrics (MRR 0.75, Precision@5 0.77) with
qualitative clinical accuracy scoring. The LLM judge has no access to the pipeline's
internal scoring — avoiding self-evaluation bias inherent in LLM-evaluates-LLM setups.

Usage:
    export GROQ_API_KEY=your_key_here
    python evaluate_llm_judge.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from pipeline.embeddings import get_chroma_client, get_collection, index_trials, semantic_search
from pipeline.ner import extract_clinical_entities
from pipeline.scorer import score_patient_trial, rank_trials
from pipeline.parser import parse_patient

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K = 5           # Number of top retrieved trials to judge per patient
SLEEP_BETWEEN = 0.5 # Seconds between Groq calls to respect rate limits
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Judge Prompt ──────────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are a clinical oncology expert reviewing whether a breast cancer clinical trial is appropriate for a specific patient.

Patient Profile:
{patient_summary}

Clinical Trial:
Title: {trial_title}
Conditions: {trial_conditions}
Interventions: {trial_interventions}
Eligibility Criteria: {trial_eligibility}

Evaluate this patient-trial match on two dimensions:

1. Clinical Relevance (1-5): How clinically appropriate is this trial for the patient's condition?
   1 = Not relevant at all
   2 = Marginally relevant
   3 = Moderately relevant
   4 = Highly relevant
   5 = Perfect clinical match

2. Eligibility Alignment (1-5): How well does this patient appear to meet the trial's eligibility criteria?
   1 = Clearly ineligible
   2 = Likely ineligible
   3 = Uncertain eligibility
   4 = Likely eligible
   5 = Clearly eligible

Important: You are providing a qualitative clinical assessment only.
Final eligibility determinations require full clinical review by a qualified physician.

Respond in this exact JSON format with no extra text:
{{
  "clinical_relevance": <integer 1-5>,
  "eligibility_alignment": <integer 1-5>,
  "reasoning": "<one concise sentence>"
}}"""


# ── Groq Client ───────────────────────────────────────────────────────────────
def get_groq_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set.\n"
            "Run: export GROQ_API_KEY=your_key_here"
        )
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


# ── LLM Judge ────────────────────────────────────────────────────────────────
def evaluate_with_llm(client: OpenAI, patient_summary: str, trial: dict) -> dict | None:
    """
    Send a single patient-trial pair to Llama 4 Scout for clinical quality assessment.
    Returns dict with clinical_relevance, eligibility_alignment, reasoning — or None on error.
    """
    prompt = JUDGE_PROMPT.format(
        patient_summary=patient_summary,
        trial_title=trial.get("title", "N/A"),
        trial_conditions=trial.get("conditions", "N/A"),
        trial_interventions=trial.get("interventions", "N/A"),
        trial_eligibility=(trial.get("eligibility", "N/A") or "N/A")[:600]
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=250,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "clinical_relevance":    max(1, min(5, int(result.get("clinical_relevance", 0)))),
            "eligibility_alignment": max(1, min(5, int(result.get("eligibility_alignment", 0)))),
            "reasoning":             result.get("reasoning", "").strip()
        }
    except Exception as e:
        print(f"    [LLM error] {e}")
        return None


# ── Data Loading ──────────────────────────────────────────────────────────────
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


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_top_k(raw_patient: dict, trials_lookup: dict, collection, top_k: int = TOP_K):
    parsed = parse_patient(raw_patient)
    query_text = parsed.get("raw_summary", raw_patient.get("summary", ""))
    patient_entities = extract_clinical_entities(query_text)
    candidates = semantic_search(query_text, collection, top_k=50)

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
            patient_entities=patient_entities,
            trial_entities=trial_entities,
        )
        scored.append(result)

    ranked = rank_trials(scored, top_k=top_k)
    return [r["trial_id"] for r in ranked]


# ── Main Evaluation ───────────────────────────────────────────────────────────
def evaluate():
    print("Loading patients and trials...")
    raw_patients, trials = load_data()

    trials_lookup = {}
    for t in trials:
        tid = t.get("trial_id") or t.get("nct_id") or t.get("NCTId", "")
        t["trial_id"] = tid
        trials_lookup[tid] = t

    print("Indexing trials into ChromaDB...")
    trial_texts = [
        f"{t.get('conditions', '')} {t.get('interventions', '')} {t.get('eligibility', '')} {t.get('title', '')}"
        for t in trials
    ]
    chroma_client = get_chroma_client()
    index_trials(trials, trial_texts)
    collection = get_collection(chroma_client)

    print(f"Initializing LLM judge: {MODEL} via Groq\n")
    groq_client = get_groq_client()

    all_relevance  = []
    all_alignment  = []
    patient_results = []

    for raw_p in raw_patients:
        pid = raw_p["patient_id"]
        summary = raw_p.get("summary", "")
        print(f"── {pid} " + "─" * 40)

        top_ids = retrieve_top_k(raw_p, trials_lookup, collection)
        p_rel, p_align = [], []
        trial_evals = []

        for trial_id in top_ids:
            trial = trials_lookup.get(trial_id)
            if not trial:
                continue

            print(f"  Judging {trial_id[:20]}...")
            result = evaluate_with_llm(groq_client, summary, trial)
            time.sleep(SLEEP_BETWEEN)

            if result:
                p_rel.append(result["clinical_relevance"])
                p_align.append(result["eligibility_alignment"])
                trial_evals.append({
                    "trial_id": trial_id,
                    "title":    trial.get("title", "")[:80],
                    **result
                })
                print(f"    Relevance {result['clinical_relevance']}/5 | "
                      f"Alignment {result['eligibility_alignment']}/5 | "
                      f"{result['reasoning']}")

        if p_rel:
            avg_r = sum(p_rel)   / len(p_rel)
            avg_a = sum(p_align) / len(p_align)
            all_relevance.extend(p_rel)
            all_alignment.extend(p_align)
            print(f"  Patient avg → Relevance {avg_r:.2f}/5 | Alignment {avg_a:.2f}/5")
            patient_results.append({
                "patient_id":              pid,
                "summary_excerpt":         summary[:200],
                "top_trials_evaluated":    trial_evals,
                "avg_clinical_relevance":  round(avg_r, 2),
                "avg_eligibility_alignment": round(avg_a, 2),
            })

    # ── Save Results ──────────────────────────────────────────────────────────
    base = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base, "data/llm_judge_results.json")
    with open(output_path, "w") as f:
        json.dump(patient_results, f, indent=2)

    # ── Summary Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("LLM-AS-JUDGE EVALUATION RESULTS")
    print("=" * 55)
    print(f"Judge model               : Llama 4 Scout (Groq)")
    print(f"Patients evaluated        : {len(patient_results)}")
    print(f"Trials judged per patient : Top {TOP_K}")
    print(f"Total patient-trial pairs : {len(all_relevance)}")

    if all_relevance:
        avg_r_global = sum(all_relevance) / len(all_relevance)
        avg_a_global = sum(all_alignment) / len(all_alignment)
        high_rel = sum(1 for s in all_relevance if s >= 4)
        high_ali = sum(1 for s in all_alignment if s >= 4)

        print(f"\nAvg Clinical Relevance    : {avg_r_global:.2f} / 5.00")
        print(f"Avg Eligibility Alignment : {avg_a_global:.2f} / 5.00")
        print(f"High relevance (≥4/5)     : {high_rel}/{len(all_relevance)} pairs ({100*high_rel/len(all_relevance):.0f}%)")
        print(f"High alignment (≥4/5)     : {high_ali}/{len(all_alignment)} pairs ({100*high_ali/len(all_alignment):.0f}%)")

    print(f"\nFull results saved to     : data/llm_judge_results.json")
    print("=" * 55)
    print("\nNote: Scores reflect qualitative clinical assessment by an independent")
    print("LLM judge. Final eligibility determinations require physician review.")


if __name__ == "__main__":
    evaluate()
