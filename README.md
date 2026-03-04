# 🧬 Clinical Trial Navigator

**AI-powered clinical trial discovery system that matches patients to relevant trials using NLP — semantic embeddings, clinical named entity recognition, and LLM-powered explanations.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)
![spaCy](https://img.shields.io/badge/spaCy-NER-yellow)

---

## Architecture

The system runs a **three-layer NLP matching pipeline** for each patient:

### Layer 1 — Semantic Search (Sentence Transformers + ChromaDB)
Each clinical trial's text (title, conditions, interventions, eligibility) is embedded into a 384-dimensional vector using `all-MiniLM-L6-v2` and stored in ChromaDB. When a patient is selected, their clinical profile is embedded with the same model and the top 30 most semantically similar trials are retrieved via cosine similarity. This catches matches that keyword search misses — e.g., "HER2-positive metastatic carcinoma" matching "advanced HER2+ breast cancer."

### Layer 2 — Clinical Named Entity Recognition (spaCy + Domain Rules)
Both patient summaries and trial descriptions are processed through a clinical NER pipeline that extracts biomarkers (HER2, ER, BRCA), cancer types, treatments, and disease stages. Entity overlap between patient and trial is scored — shared biomarkers and matching cancer subtypes receive the highest weight.

### Layer 3 — Rule-Based Clinical Signals
Preserved from the original matching logic: disease relevance, malignancy confirmation, receptor status alignment, metastatic match, pregnancy exclusion (hard disqualifier), and age availability. These signals encode domain knowledge that pure semantic similarity might miss.

### Composite Scoring
Final match score = **50% semantic similarity + 30% rule-based signals + 20% NER entity overlap**. Trials are ranked by composite score and the top 10 matches are presented with match percentages, score breakdowns, and explanations.

### LLM-Powered Explanations & Chat (Groq — Llama 3.3 70B)
Each match includes an LLM-generated clinician-friendly explanation. A RAG chat interface lets users ask follow-up questions grounded in the patient profile and matched trials.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Vector Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| NER | spaCy + custom clinical entity rules |
| Rule Engine | Domain-specific clinical signal scoring |
| LLM | Groq API (Llama 3.3 70B) |
| Data Source | ClinicalTrials.gov (public), synthetic patients |

---

## Project Structure

```
clinical-trial-navigator/
├── app.py                  # Streamlit UI
├── config.py               # Constants, model names, scoring weights
├── pipeline/
│   ├── parser.py           # Patient profile extraction, trial text prep
│   ├── ner.py              # Clinical NER (spaCy + domain patterns)
│   ├── embeddings.py       # Sentence transformer + ChromaDB operations
│   ├── scorer.py           # Three-layer composite scoring
│   └── narrative.py        # Groq LLM explanations + RAG chat
├── data/
│   ├── patients/           # Synthetic patient profiles
│   └── trials/             # Clinical trial data from ClinicalTrials.gov
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/clinical-trial-navigator.git
cd clinical-trial-navigator
pip install -r requirements.txt
export GROQ_API_KEY="your-key"
streamlit run app.py
```

> **First run:** The embedding model (~80MB) downloads once from HuggingFace, then is cached.

---

## Disclaimer

This tool is for **educational and research purposes only**. It does not make medical or eligibility determinations. Always consult qualified healthcare professionals for clinical decisions.

---

## License

MIT
