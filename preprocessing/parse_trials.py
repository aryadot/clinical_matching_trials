import pandas as pd
import re
from pathlib import Path
import json


# Load trial CSV
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "trials" / "breast_cancer_trials.csv"
OUTPUT_PATH = BASE_DIR / "data" / "trials" / "parsed_trials.json"

df = pd.read_csv(DATA_PATH)


def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_eligibility(text):
    """
    Splits eligibility text into inclusion and exclusion blocks.
    Returns lists of inclusion and exclusion criteria.
    """
    text = clean_text(text)

    inclusion = []
    exclusion = []

    # Case-insensitive split
    inc_match = re.split(r"inclusion criteria\s*[:\-]?", text, flags=re.I)
    exc_match = re.split(r"exclusion criteria\s*[:\-]?", text, flags=re.I)

    if len(inc_match) > 1:
        inclusion_block = inc_match[1]
    else:
        inclusion_block = ""

    if len(exc_match) > 1:
        exclusion_block = exc_match[1]
        # Remove inclusion text accidentally captured
        if inclusion_block and exclusion_block in inclusion_block:
            inclusion_block = inclusion_block.replace(exclusion_block, "")
    else:
        exclusion_block = ""

    inclusion = extract_bullets(inclusion_block)
    exclusion = extract_bullets(exclusion_block)

    return inclusion, exclusion


def extract_bullets(text):
    """
    Converts blocks of text into individual criteria.
    """
    if not text:
        return []

    lines = text.split("\n")
    criteria = []

    for line in lines:
        line = line.strip(" -*•\t")
        if len(line) > 5:
            criteria.append(line)

    return criteria



# Parse all trials


parsed_trials = []

for _, row in df.iterrows():
    eligibility_text = row.get("Eligibility Criteria", "")
    inclusion, exclusion = split_eligibility(eligibility_text)

    parsed_trials.append({
        "trial_id": row.get("NCT Number"),
        "title": row.get("Study Title"),
        "status": row.get("Study Status"),
        "conditions": row.get("Conditions"),
        "interventions": row.get("Interventions"),
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion
    })

# Save parsed output


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(parsed_trials, f, indent=2)

print(f"Parsed {len(parsed_trials)} trials → {OUTPUT_PATH}")
