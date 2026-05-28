"""
evaluate_llm_judge.py — LLM-as-Reranker Evaluation for Clinical Trial Navigator

Redesigned evaluation: instead of scoring each trial independently, the LLM
sees ALL top-20 retrieved trials simultaneously and ranks them from most to least
relevant for the patient. This produces relative judgments within the available
options rather than absolute clinical quality scores against an imaginary ideal trial.

This mirrors how production clinical AI systems use LLMs for evaluation — the
question is not "is this a perfect match?" but "given these options, which are
the best matches for this patient?"

Metrics reported:
  - Rank correlation between LLM ranking and pipeline ranking (Spearman)
  - Agreement@k: how often LLM top-k overlaps with pipeline top-k
  - Per-patient LLM top trial with reasoning


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
TOP_K_RETRIEVE = 20   # Show LLM all top 20 trials
TOP_K_REPORT   = 5    # Report agreement on top 5
SLEEP_BETWEEN  = 1.0  # Seconds between Groq calls
MODEL          = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Reranker Prompt ───────────────────────────────────────────────────────────
RERANKER_PROMPT = """You are a clinical oncology expert. A patient matching system has retrieved {n_trials} breast cancer clinical trials as potential matches for the following patient. These are the ONLY available trials in the database.

Patient Profile:
{patient_summary}

Retrieved Clinical Trials:
{trial_list}

Your task: Rank these {n_trials} trials from most to least clinically appropriate for this patient, considering both clinical relevance and likely eligibility. These are the only options available — rank them relative to each other.

For your top 3 ranked trials, provide a one-sentence reasoning.

Respond in this exact JSON format with no extra text:
{{
  "ranking": [<trial_number_1>, <trial_number_2>, ..., <trial_number_{n_trials}>],
  "reasoning": {{
    "1": "<one sentence for your #1 choice>",
    "2": "<one sentence for your #2 choice>",
    "3": "<one sentence for your #3 choice>"
  }}
}}

Where ranking is a list of trial numbers (1 to {n_trials}) ordered from best to worst match."""


# ── Groq Client ───────────────────────────────────────────────────────────────
def get_groq_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Run: export GROQ_API_KEY=your_key_here")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


# ── LLM Reranker ──────────────────────────────────────────────────────────────
def rerank_with_llm(client: OpenAI, patient_summary: str, trials: list) -> dict | None:
    """
    Show LLM all retrieved trials simultaneously and ask for relative ranking.
    Returns dict with ranking list and reasoning for top 3.
    """
    # Build numbered trial list
    trial_list_text = ""
    for i, trial in enumerate(trials, 1):
        eligibility = (trial.get("eligibility", "") or "")[:300]
        trial_list_text += (
            f"\nTrial {i}:\n"
            f"  Title: {trial.get('title', 'N/A')[:80]}\n"
            f"  Conditions: {trial.get('conditions', 'N/A')[:100]}\n"
            f"  Interventions: {trial.get('interventions', 'N/A')[:100]}\n"
            f"  Eligibility (excerpt): {eligibility}\n"
        )

    prompt = RERANKER_PROMPT.format(
        n_trials=len(trials),
        patient_summary=patient_summary[:400],
        trial_list=trial_list_text,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        ranking  = result.get("ranking", [])
        reasoning = result.get("reasoning", {})

        # Validate ranking is a permutation of 1..n
        expected = set(range(1, len(trials) + 1))
        if set(ranking) != expected:
            return None

        return {"ranking": ranking, "reasoning": reasoning}

    except Exception as e:
        print(f"    [LLM error] {e}")
        return None


# ── Spearman rank correlation ─────────────────────────────────────────────────
def spearman_correlation(rank1: list, rank2: list) -> float:
    """Compute Spearman rank correlation between two rankings."""
    n = len(rank1)
    if n < 2:
        return 1.0
    d_sq = sum((r1 - r2) ** 2 for r1, r2 in zip(rank1, rank2))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


# ── Agreement@k ──────────────────────────────────────────────────────────────
def agreement_at_k(llm_ranking: list, k: int) -> float:
    """
    Fraction of LLM top-k trials that also appear in pipeline top-k.
    Pipeline top-k is trials[0:k] since they are already ranked by composite score.
    LLM ranking is 1-indexed trial numbers.
    """
    pipeline_top_k = set(range(1, k + 1))  # Trials 1..k are pipeline top-k
    llm_top_k      = set(llm_ranking[:k])
    return len(pipeline_top_k & llm_top_k) / k


# ── Data loading ──────────────────────────────────────────────────────────────
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
def retrieve_top_k(raw_patient, trials_lookup, collection, top_k=TOP_K_RETRIEVE):
    parsed          = parse_patient(raw_patient)
    query_text      = parsed.get("raw_summary", raw_patient.get("summary", ""))
    patient_entities = extract_clinical_entities(query_text)
    candidates      = semantic_search(query_text, collection, top_k=50)
    scored = []
    for c in candidates:
        trial = trials_lookup.get(c["trial_id"])
        if not trial:
            continue
        trial_entities = extract_clinical_entities(
            f"{trial.get('conditions', '')} {trial.get('interventions', '')} {trial.get('eligibility', '')}"
        )
        result = score_patient_trial(
            patient=parsed, trial=trial,
            semantic_similarity=c["similarity"],
            patient_entities=patient_entities,
            trial_entities=trial_entities,
        )
        scored.append(result)
    ranked = rank_trials(scored, top_k=top_k)
    return [trials_lookup[r["trial_id"]] for r in ranked if r["trial_id"] in trials_lookup]


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate():
    print("Loading data...")
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
    client_chroma = get_chroma_client()
    index_trials(trials, trial_texts)
    collection = get_collection(client_chroma)

    print(f"Initializing LLM reranker: {MODEL} via Groq\n")
    groq_client = get_groq_client()

    all_spearman     = []
    all_agreement_1  = []
    all_agreement_3  = []
    all_agreement_5  = []
    patient_results  = []

    for raw_p in raw_patients:
        pid     = raw_p["patient_id"]
        summary = raw_p.get("summary", "")
        print(f"── {pid} " + "─" * 40)

        top_trials = retrieve_top_k(raw_p, trials_lookup, collection)
        if not top_trials:
            print(f"  No trials retrieved, skipping")
            continue

        print(f"  Retrieved {len(top_trials)} trials. Sending to LLM reranker...")
        result = rerank_with_llm(groq_client, summary, top_trials)
        time.sleep(SLEEP_BETWEEN)

        if not result:
            print(f"  LLM reranking failed, skipping")
            continue

        llm_ranking = result["ranking"]
        reasoning   = result["reasoning"]

        # Pipeline ranking is 1..n (already sorted by composite score)
        pipeline_ranking = list(range(1, len(top_trials) + 1))

        # Spearman correlation between pipeline and LLM ranking
        spearman = spearman_correlation(pipeline_ranking, llm_ranking)
        all_spearman.append(spearman)

        # Agreement@k
        agr1 = agreement_at_k(llm_ranking, 1)
        agr3 = agreement_at_k(llm_ranking, min(3, len(top_trials)))
        agr5 = agreement_at_k(llm_ranking, min(5, len(top_trials)))
        all_agreement_1.append(agr1)
        all_agreement_3.append(agr3)
        all_agreement_5.append(agr5)

        # LLM top choice
        llm_top_idx  = llm_ranking[0] - 1
        llm_top_trial = top_trials[llm_top_idx] if llm_top_idx < len(top_trials) else {}

        print(f"  Spearman correlation: {spearman:.3f}")
        print(f"  Agreement@1: {agr1:.0%} | @3: {agr3:.0%} | @5: {agr5:.0%}")
        print(f"  LLM top choice: {llm_top_trial.get('trial_id', 'N/A')}")
        print(f"  Reasoning: {reasoning.get('1', 'N/A')}")

        patient_results.append({
            "patient_id":        pid,
            "summary_excerpt":   summary[:200],
            "n_trials_shown":    len(top_trials),
            "spearman":          round(spearman, 3),
            "agreement_at_1":    agr1,
            "agreement_at_3":    agr3,
            "agreement_at_5":    agr5,
            "llm_top_trial":     llm_top_trial.get("trial_id", ""),
            "llm_ranking":       llm_ranking,
            "llm_reasoning":     reasoning,
        })

    # Save results
    base = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base, "data/llm_reranker_results.json")
    with open(output_path, "w") as f:
        json.dump(patient_results, f, indent=2)

    # Summary
    print("\n" + "=" * 55)
    print("LLM-AS-RERANKER EVALUATION RESULTS")
    print("=" * 55)
    print(f"Reranker model         : Llama 4 Scout (Groq)")
    print(f"Patients evaluated     : {len(patient_results)}")
    print(f"Trials shown per patient: Top {TOP_K_RETRIEVE}")

    if all_spearman:
        print(f"\nAvg Spearman correlation : {sum(all_spearman)/len(all_spearman):.3f}")
        print(f"Agreement@1              : {sum(all_agreement_1)/len(all_agreement_1):.0%}")
        print(f"Agreement@3              : {sum(all_agreement_3)/len(all_agreement_3):.0%}")
        print(f"Agreement@5              : {sum(all_agreement_5)/len(all_agreement_5):.0%}")

    print(f"\nFull results saved to  : data/llm_reranker_results.json")
    print("=" * 55)
    print("\nNote: Rankings are relative — LLM selects best available matches")
    print("from the retrieved set, not from all possible trials worldwide.")


if __name__ == "__main__":
    evaluate()