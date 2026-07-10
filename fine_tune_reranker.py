"""
fine_tune_reranker.py — Fine Tune the Cross Encoder Reranker

Starting point: cross-encoder/ms-marco-MiniLM-L-6-v2, a general purpose
passage reranker with no clinical trial knowledge.

This script fine tunes it on patient-trial pairs so it learns what makes
a breast cancer trial actually relevant to a specific patient, instead of
relying on generic semantic overlap.

Labels come from the same rule based scorer used to build ground truth
in evaluate_retrieval.py (compute_rule_score) Positive pairs are patient-trial matches the rule engine scores
as relevant. Negative pairs are randomly sampled non matches.

"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

from pipeline.scorer import compute_rule_score
from pipeline.parser import parse_patient

BASE_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR       = "models/reranker-finetuned"
RELEVANCE_THRESHOLD = 0.55
NEGATIVES_PER_POSITIVE = 3
EPOCHS           = 4
BATCH_SIZE       = 16
TEST_SPLIT       = 0.2


def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "data/patients/synthetic_patients.json")) as f:
        patients = json.load(f)
    with open(os.path.join(base, "data/trials/parsed_trials.json")) as f:
        trials = json.load(f)
    return patients, trials


def trial_text(trial: dict) -> str:
    eligibility = trial.get("eligibility", "") or ""
    if not eligibility and trial.get("inclusion_criteria"):
        eligibility = " ".join(trial["inclusion_criteria"])
    return f"{trial.get('title', '')}. {trial.get('conditions', '')}. {eligibility[:500]}".strip()


def build_training_pairs(patients, trials):
    """
    Builds labeled (patient_summary, trial_text, label) pairs.
    Label 1 = rule engine considers this trial relevant to the patient.
    Label 0 = randomly sampled trial the rule engine does not consider relevant.
    """
    pairs = []

    for raw_p in patients:
        patient = parse_patient(raw_p)
        patient_summary = raw_p["summary"]

        relevant_trials = []
        irrelevant_trials = []

        for trial in trials:
            rule_score, _ = compute_rule_score(patient, trial)
            if rule_score >= RELEVANCE_THRESHOLD:
                relevant_trials.append(trial)
            else:
                irrelevant_trials.append(trial)

        for trial in relevant_trials:
            pairs.append(InputExample(texts=[patient_summary, trial_text(trial)], label=1.0))

        # Sample negatives so the dataset isn't overwhelmingly one sided
        num_negatives = min(len(relevant_trials) * NEGATIVES_PER_POSITIVE, len(irrelevant_trials))
        sampled_negatives = random.sample(irrelevant_trials, num_negatives) if num_negatives > 0 else []
        for trial in sampled_negatives:
            pairs.append(InputExample(texts=[patient_summary, trial_text(trial)], label=0.0))

        print(f"  {raw_p['patient_id']}: {len(relevant_trials)} positive, {len(sampled_negatives)} negative")

    return pairs


def main():
    random.seed(42)

    print("Loading patients and trials...")
    patients, trials = load_data()

    print("\nBuilding labeled training pairs from rule based ground truth...")
    pairs = build_training_pairs(patients, trials)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * (1 - TEST_SPLIT))
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Train: {len(train_pairs)}  Test: {len(test_pairs)}")

    print(f"\nLoading base model: {BASE_MODEL}")
    model = CrossEncoder(BASE_MODEL, num_labels=1)

    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=BATCH_SIZE)

    evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_pairs, name="reranker-test")

    print(f"\nFine tuning for {EPOCHS} epochs...")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=OUTPUT_DIR,
        show_progress_bar=True,
    )

    print(f"\nFine tuned model saved to {OUTPUT_DIR}")
    print("Update CROSS_ENCODER_MODEL in pipeline/scorer.py to this path to use it in the app.")


if __name__ == "__main__":
    main()
