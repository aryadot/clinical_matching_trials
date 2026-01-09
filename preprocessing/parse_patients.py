import json
import re
from pathlib import Path


# Resolve project root
BASE_DIR = Path(__file__).resolve().parents[1]
PATIENT_PATH = BASE_DIR / "data" / "patients" / "synthetic_patients.json"
OUTPUT_PATH = BASE_DIR / "data" / "patients" / "parsed_patients.json"



def extract_age(text):
    match = re.search(r"(\d{2})[- ]?year[- ]?old", text.lower())
    return int(match.group(1)) if match else "unknown"


def extract_pregnancy(text):
    keywords = ["pregnan", "gestation", "trimester", "intrauterine"]
    return any(k in text.lower() for k in keywords)


def extract_receptor_status(text):
    status = {"ER": "unknown", "HER2": "unknown"}

    if "er-positive" in text.lower() or "er positive" in text.lower():
        status["ER"] = "positive"
    if "her2-positive" in text.lower() or "her2 positive" in text.lower():
        status["HER2"] = "positive"
    if "her2-low" in text.lower():
        status["HER2"] = "low"
    if "her2-negative" in text.lower() or "her2 negative" in text.lower():
        status["HER2"] = "negative"

    return status


def extract_metastatic_status(text):
    if "metastatic" in text.lower():
        return True
    return False


def extract_prior_cancer(text):
    keywords = ["history of breast cancer", "prior breast cancer", "second primary"]
    return any(k in text.lower() for k in keywords)


def extract_prior_therapy(text):
    keywords = ["chemotherapy", "endocrine therapy", "targeted therapy", "trastuzumab"]
    return any(k in text.lower() for k in keywords)


def extract_comorbidities(text):
    keywords = ["autoimmune", "heart failure", "cardiac", "diabetes"]
    found = [k for k in keywords if k in text.lower()]
    return found if found else []


# -----------------------------
# Load patients
# -----------------------------
with open(PATIENT_PATH) as f:
    patients = json.load(f)

parsed_patients = []

for patient in patients:
    summary = patient["summary"]

    parsed = {
        "patient_id": patient["patient_id"],
        "age": extract_age(summary),
        "pregnant": extract_pregnancy(summary),
        "receptor_status": extract_receptor_status(summary),
        "metastatic": extract_metastatic_status(summary),
        "prior_cancer_history": extract_prior_cancer(summary),
        "prior_systemic_therapy": extract_prior_therapy(summary),
        "comorbidities": extract_comorbidities(summary),
        "raw_summary": summary
    }

    parsed_patients.append(parsed)



OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(parsed_patients, f, indent=2)

print(f"Parsed {len(parsed_patients)} patients → {OUTPUT_PATH}")
