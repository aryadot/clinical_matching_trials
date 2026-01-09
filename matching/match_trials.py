import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

TRIALS_PATH = BASE_DIR / "data" / "trials" / "parsed_trials.json"
PATIENTS_PATH = BASE_DIR / "data" / "patients" / "parsed_patients.json"
OUTPUT_PATH = BASE_DIR / "data" / "top_matches.json"


def score_trial(patient, trial):
    score = 0
    reasons = []

    inclusion_text = " ".join(trial.get("inclusion_criteria", [])).lower()
    exclusion_text = " ".join(trial.get("exclusion_criteria", [])).lower()
    summary = patient["raw_summary"].lower()

    # Core disease relevance
    if "breast cancer" in summary:
        score += 2
        reasons.append("Breast cancer focus")

    # Age signal
    if patient["age"] != "unknown":
        score += 1
        reasons.append("Age information available")

    # Pregnancy hard exclusion
    if patient["pregnant"] and "pregnan" in exclusion_text:
        return 0, ["Pregnancy exclusion"]

    # Malignancy check
    if "benign" not in summary:
        score += 2
        reasons.append("Confirmed malignancy")

    # Metastatic relevance
    if patient["metastatic"] and "metastatic" in inclusion_text:
        score += 1
        reasons.append("Metastatic disease aligns")

    # Receptor matching
    receptors = patient["receptor_status"]

    if receptors["HER2"] != "unknown" and receptors["HER2"] in inclusion_text:
        score += 2
        reasons.append("HER2 status aligns")

    if receptors["ER"] != "unknown" and receptors["ER"] in inclusion_text:
        score += 1
        reasons.append("ER status aligns")

    return score, reasons


def main():
    with open(TRIALS_PATH) as f:
        trials = json.load(f)

    with open(PATIENTS_PATH) as f:
        patients = json.load(f)

    results = {}

    for patient in patients:
        scored_trials = []

        for trial in trials:
            score, reasons = score_trial(patient, trial)
            if score > 0:
                scored_trials.append({
                    "trial_id": trial["trial_id"],
                    "title": trial["title"],
                    "raw_score": score,
                    "reasons": reasons
                })

        if not scored_trials:
            results[patient["patient_id"]] = []
            continue

        max_score = max(t["raw_score"] for t in scored_trials)

        for t in scored_trials:
            t["match_percentage"] = round((t["raw_score"] / max_score) * 100, 1)

        top_trials = sorted(
            scored_trials,
            key=lambda x: x["match_percentage"],
            reverse=True
        )[:5]

        results[patient["patient_id"]] = top_trials

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated top 5 trial matches → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
