"""
config.py — Constants, model names, and scoring weights
"""
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PATIENTS_RAW = DATA_DIR / "patients" / "synthetic_patients.json"
PATIENTS_PARSED = DATA_DIR / "patients" / "parsed_patients.json"
TRIALS_CSV = DATA_DIR / "trials" / "breast_cancer_trials.csv"
TRIALS_PARSED = DATA_DIR / "trials" / "parsed_trials.json"

# ── Models ──
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# ── ChromaDB ──
CHROMA_COLLECTION = "clinical_trials"

# NOTE: composite ranking weights (semantic 0.35 / cross-encoder 0.65) live
# in pipeline/scorer.py, not here — that is the single source of truth.
# Rule-based scoring below is used only as a hard-exclusion gate, never as
# a weighted component of the composite score.

# ── Rule-based signal scores (from original project, preserved) ──
SIGNAL_DISEASE_MATCH = 2
SIGNAL_MALIGNANCY = 2
SIGNAL_RECEPTOR_MATCH = 2
SIGNAL_METASTATIC_MATCH = 1
SIGNAL_AGE_AVAILABLE = 1
SIGNAL_ER_MATCH = 1
PENALTY_PREGNANCY = -999    # Hard exclusion

# ── Clinical Entity Keywords ──
BIOMARKERS = [
    "HER2", "ER", "PR", "BRCA1", "BRCA2", "PD-L1", "Ki-67",
    "PIK3CA", "ESR1", "EGFR", "ALK", "ROS1", "NTRK",
    "HER2-positive", "HER2-negative", "HER2-low",
    "ER-positive", "ER-negative", "PR-positive", "PR-negative",
    "triple-negative", "hormone receptor",
]

CANCER_TYPES = [
    "breast cancer", "invasive ductal carcinoma", "lobular carcinoma",
    "ductal carcinoma in situ", "DCIS", "TNBC", "triple negative",
    "metastatic breast cancer", "early-stage breast cancer",
    "HER2-positive breast cancer", "inflammatory breast cancer",
    "mammary cancer", "breast neoplasm",
]

TREATMENTS = [
    "trastuzumab", "pertuzumab", "chemotherapy", "endocrine therapy",
    "tamoxifen", "letrozole", "anastrozole", "fulvestrant",
    "paclitaxel", "doxorubicin", "cyclophosphamide", "capecitabine",
    "pembrolizumab", "atezolizumab", "nivolumab", "immunotherapy",
    "radiation", "radiotherapy", "surgery", "mastectomy", "lumpectomy",
    "T-DXd", "trastuzumab deruxtecan", "sacituzumab", "tucatinib",
    "CDK4/6 inhibitor", "palbociclib", "ribociclib", "abemaciclib",
]

STAGES = [
    "stage I", "stage II", "stage III", "stage IV",
    "metastatic", "locally advanced", "early-stage",
    "recurrent", "refractory", "unresectable",
]
