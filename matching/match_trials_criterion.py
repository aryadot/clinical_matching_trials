import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

TRIALS_PATH = BASE_DIR / "data" / "trials" / "parsed_trials.json"
PATIENTS_PATH = BASE_DIR / "data" / "patients" / "parsed_patients.json"
OUTPUT_PATH = BASE_DIR / "data" / "matches_criterion_level.json"


# -----------------------------
# Utility helpers
# -----------------------------

def text_contains_any(text, keywords):
    t = text.lower()
    return any(k in t for k in keywords)


def normalize(text):
    return text.lower().strip()


# -----------------------------
# Criterion evaluation
# -----------------------------

def evaluate_inclusion(criterion, patient):
    c = normalize(criterion)
    summary = patient["raw_summary"].lower()

    # Age
    if "age" in c or "years" in c or "≥18" in c or "18 years" in c:
        if patient["age"] != "unknown":
            return "Met", "Age present in patient summary"
        return "Unknown", "Age not specified"

    # Pregnancy
    if "pregnan" in c:
        if patient["pregnant"]:
            return "Met", "Patient is pregnant"
        return "Not Met", "Patient is not pregnant"

    # Metastatic
    if "metastatic" in c:
        if patient["metastatic"]:
            return "Met", "Metastatic disease mentioned"
        return "Not Met", "No metastatic disease mentioned"

    # ER status
    if "er-positive" in c or "estrogen receptor" in c:
        if patient["receptor_status"]["ER"] == "positive":
            return "Met", "ER-positive mentioned"
        if patient["receptor_status"]["ER"] == "unknown":
            return "Unknown", "ER status not specified"
        return "Not Met", "ER-negative"

    # HER2 status
    if "her2" in c:
        her2 = patient["receptor_status"]["HER2"]
        if her2 != "unknown" and her2 in c:
            return "Met", f"HER2 status aligns ({her2})"
        if her2 == "unknown":
            return "Unknown", "HER2 status not specified"
        return "Not Met", f"HER2 status mismatch ({her2})"

    # Default
    if text_contains_any(summary, c.split()):
        return "Met", "Relevant terms found in patient summary"

    return "Unknown", "Insufficient information"


def evaluate_exclusion(criterion, patient):
    c = normalize(criterion)
    summary = patient["raw_summary"].lower()

    # Pregnancy exclusion
    if "pregnan" in c:
        if patient["pregnant"]:
            return "Met", "Pregnancy exclusion applies"
        return "Not Met", "Patient not pregnant"

    # Prior therapy exclusion
    if "prior" in c or "previous" in c:
        if patient["prior_systemic_therapy"]:
            return "Met", "Prior systemic therapy present"
        return "Not Met", "No prior systemic therapy"

    # Comorbidity exclusions
    if text_contains_any(c, ["autoimmune", "cardiac", "heart failure", "diabetes"]):
        if patient["comorbidities"]:
            return "Met", "Relevant comorbidity present"
        return "Not Met", "No relevant comorbidity"

    # Default
    if text_contains_any(summary, c.split()):
        return "Met", "Exclusion condition mentioned"

    return "Unknown", "Insufficient information"


# -----------------------------
# Main matching loop
# -----------------------------

def main():
    with open(TRIALS_PATH) as f:
        trials = json.load(f)

    with open(PATIENTS_PATH) as f:
        patients = json.load(f)

    results = {}

    for patient in patients:
        patient_results = []

        for trial in trials:
            inclusion_results = []
            exclusion_results = []

            # Evaluate inclusion criteria
            for inc in trial.get("inclusion_criteria", []):
                status, evidence = evaluate_inclusion(inc, patient)
                inclusion_results.append({
                    "criterion": inc,
                    "type": "inclusion",
                    "status": status,
                    "evidence": evidence
                })

            # Evaluate exclusion criteria
            excluded = False
            for exc in trial.get("exclusion_criteria", []):
                status, evidence = evaluate_exclusion(exc, patient)
                exclusion_results.append({
                    "criterion": exc,
                    "type": "exclusion",
                    "status": status,
                    "evidence": evidence
                })
                if status == "Met":
                    excluded = True

            # Decision logic
            if excluded:
                decision = "Ineligible"
            else:
                met_inclusions = sum(1 for r in inclusion_results if r["status"] == "Met")
                if met_inclusions >= max(1, len(inclusion_results) // 2):
                    decision = "Likely Eligible"
                else:
                    decision = "Needs Review"

            patient_results.append({
                "trial_id": trial["trial_id"],
                "title": trial["title"],
                "decision": decision,
                "inclusion_evaluation": inclusion_results,
                "exclusion_evaluation": exclusion_results
            })

        results[patient["patient_id"]] = patient_results

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Criterion-level matches generated → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
