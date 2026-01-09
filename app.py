import json
import streamlit as st
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MATCHES_PATH = BASE_DIR / "data" / "top_matches.json"
PATIENTS_PATH = BASE_DIR / "data" / "patients" / "parsed_patients.json"

# -----------------------------
# Load data
# -----------------------------
with open(MATCHES_PATH) as f:
    strong_matches = json.load(f)

with open(PATIENTS_PATH) as f:
    patients = json.load(f)

patient_lookup = {p["patient_id"]: p for p in patients}

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Clinical Trial Finder",
    layout="wide"
)

st.title("🧬 Clinical Trial Finder")
st.caption(
    "Prototype that recommends high-confidence clinical trials based on patient profiles. "
    "For educational purposes only."
)

# -----------------------------
# Sidebar: patient selection
# -----------------------------
st.sidebar.header("Select Patient")
patient_id = st.sidebar.selectbox(
    "Patient ID",
    list(strong_matches.keys())
)

patient = patient_lookup[patient_id]

# -----------------------------
# Patient summary
# -----------------------------
st.subheader(f"Patient {patient_id} Summary")
st.write(patient["raw_summary"])

st.markdown("---")

# -----------------------------
# Strong matches display
# -----------------------------
matches = strong_matches.get(patient_id, [])

st.subheader("Strong Clinical Trial Matches")

if not matches:
    st.info(
        "No strong clinical trial matches were found for this patient "
        "based on the available information."
    )
else:
    for trial in matches:
        with st.expander(f"{trial['trial_id']} — {trial['title']}"):
            st.markdown(f"**Match Score:** {trial['match_percentage']}%")


            st.markdown("**Why this trial matches:**")
            for reason in trial["reasons"]:
                st.markdown(f"• {reason}")

# -----------------------------
# Footer disclaimer
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ This tool is for informational and educational purposes only and does not "
    "constitute medical advice or clinical trial eligibility determination."
)
