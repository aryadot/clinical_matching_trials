"""
enrich_trials.py — Fetch Eligibility Criteria from ClinicalTrials.gov API

Fetches real inclusion/exclusion criteria for all 1,046 trials using the
ClinicalTrials.gov v2 API and updates parsed_trials.json in place.

Usage:
    python enrich_trials.py

Saves enriched data to: data/trials/parsed_trials.json (updates in place)
Also saves a backup to:  data/trials/parsed_trials_backup.json
"""

import json
import os
import time
import requests
from pathlib import Path

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRIALS_PATH = os.path.join(BASE_DIR, "data/trials/parsed_trials.json")
BACKUP_PATH = os.path.join(BASE_DIR, "data/trials/parsed_trials_backup.json")

API_URL    = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"
SLEEP      = 0.12   # ~8 requests/sec — safely under rate limit
BATCH_SAVE = 50     # Save progress every N trials


def fetch_eligibility(nct_id: str) -> dict:
    """
    Fetch eligibility module for a single trial from ClinicalTrials.gov v2 API.
    Returns dict with eligibility_text, inclusion_criteria, exclusion_criteria,
    min_age, max_age, sex — or empty values on failure.
    """
    try:
        url = API_URL.format(nct_id=nct_id)
        params = {"fields": "EligibilityModule,StatusModule"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        elig = (data.get("protocolSection", {})
                    .get("eligibilityModule", {}))

        raw_text = elig.get("eligibilityCriteria", "")

        # Split into inclusion / exclusion blocks
        inclusion, exclusion = [], []
        if raw_text:
            lower = raw_text.lower()
            inc_idx = lower.find("inclusion criteria")
            exc_idx = lower.find("exclusion criteria")

            if inc_idx != -1 and exc_idx != -1:
                inc_block = raw_text[inc_idx:exc_idx]
                exc_block = raw_text[exc_idx:]
            elif inc_idx != -1:
                inc_block = raw_text[inc_idx:]
                exc_block = ""
            elif exc_idx != -1:
                inc_block = raw_text[:exc_idx]
                exc_block = raw_text[exc_idx:]
            else:
                inc_block = raw_text
                exc_block = ""

            # Extract bullet lines
            def extract_bullets(block: str) -> list[str]:
                lines = [l.strip().lstrip("*-•·") .strip()
                         for l in block.splitlines()
                         if len(l.strip()) > 15]
                return [l for l in lines if l]

            inclusion = extract_bullets(inc_block)
            exclusion = extract_bullets(exc_block)

        return {
            "eligibility":         raw_text[:2000] if raw_text else "",
            "inclusion_criteria":  inclusion[:15],
            "exclusion_criteria":  exclusion[:10],
            "min_age":             elig.get("minimumAge", ""),
            "max_age":             elig.get("maximumAge", ""),
            "sex":                 elig.get("sex", ""),
        }

    except Exception as e:
        return {}


def enrich():
    # Load trials
    with open(TRIALS_PATH) as f:
        trials = json.load(f)

    # Backup original
    with open(BACKUP_PATH, "w") as f:
        json.dump(trials, f)
    print(f"Backup saved to {BACKUP_PATH}")

    total       = len(trials)
    enriched    = 0
    already_has = 0
    failed      = 0

    print(f"Fetching eligibility criteria for {total} trials...\n")

    for i, trial in enumerate(trials):
        nct_id = trial.get("trial_id", "")

        # Skip if already enriched
        if trial.get("eligibility"):
            already_has += 1
            continue

        if not nct_id.startswith("NCT"):
            failed += 1
            continue

        result = fetch_eligibility(nct_id)
        if result:
            trial.update(result)
            enriched += 1
        else:
            failed += 1

        # Progress
        if (i + 1) % 10 == 0:
            pct = (i + 1) / total * 100
            print(f"  [{i+1}/{total}] {pct:.0f}% — enriched {enriched}, failed {failed}")

        # Save progress periodically
        if (i + 1) % BATCH_SAVE == 0:
            with open(TRIALS_PATH, "w") as f:
                json.dump(trials, f, indent=2)
            print(f"  Progress saved at trial {i+1}")

        time.sleep(SLEEP)

    # Final save
    with open(TRIALS_PATH, "w") as f:
        json.dump(trials, f, indent=2)

    # Summary
    with_eligibility = sum(1 for t in trials if t.get("eligibility"))
    print(f"\n{'='*50}")
    print(f"ENRICHMENT COMPLETE")
    print(f"{'='*50}")
    print(f"Total trials      : {total}")
    print(f"Newly enriched    : {enriched}")
    print(f"Already had data  : {already_has}")
    print(f"Failed / no NCT   : {failed}")
    print(f"Now have eligibility: {with_eligibility}/{total} trials")
    print(f"{'='*50}")


if __name__ == "__main__":
    enrich()
