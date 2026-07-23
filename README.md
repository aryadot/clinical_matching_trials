# 🧬 Clinical Trial Navigator

**AI-powered clinical trial discovery system that matches patients to relevant trials using NLP — semantic embeddings, clinical named entity recognition, and LLM-powered explanations.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)

---

# 🧬 Clinical Trial Navigator

**AI-powered clinical trial discovery system that matches patients to relevant trials using a weight-free, rank-fusion pipeline — semantic embeddings, keyword/entity overlap, a fine-tuned cross-encoder, and LLM-powered explanations.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)

---

## Architecture

For each patient, the pipeline runs two independent retrieval strategies, fuses them without hand-picked weights, deep-reads the survivors, then fuses again:

```
[Patient Record]
     |
     +--> Strategy A: Vector DB Search (Sentence Transformers + ChromaDB)
     |         --> List 1, ranked by semantic similarity
     +--> Strategy B: Keyword/Entity Overlap Search (curated term list)
     |         --> List 2, ranked by Sørensen-Dice Index
     v
1. TWO-WAY RECIPROCAL RANK FUSION (RRF) --> Top-50 Master List
     v
2. Rule-based hard-exclusion guardrail (pregnancy / age / exclusion
   criteria) — deterministic, disqualifies before scoring continues
     v
3. Cross-Encoder Stage (fine-tuned cross-encoder/ms-marco-MiniLM-L-6-v2)
   deep-reads each surviving patient-trial pair
     v
4. DUAL-AXIS FINAL RRF (Master-list rank × Cross-Encoder rank)
     v
Eligibility % (rescaled relative to the top trial) shown to the user
```

**Why rank fusion instead of weighted scoring:** an earlier version of this
pipeline combined signals with a hand-picked weighted sum (e.g. 35%
semantic + 65% cross-encoder). Those weights were never validated against
labeled ground truth — there is no clinician-confirmed "correct match"
dataset backing this project. Reciprocal Rank Fusion sidesteps that
problem: it only ever combines *rank positions*, never raw scores, so
there's no weight to justify and no cross-scale comparison to defend.

### Strategy A — Semantic Search (Sentence Transformers + ChromaDB)
Trial text (title, conditions, interventions, eligibility) is embedded with
`all-MiniLM-L6-v2` and stored in ChromaDB. The patient's clinical summary
is embedded the same way and compared via cosine similarity.

### Strategy B — Keyword/Entity Overlap Search
**Not spaCy, not a trained NER model.** This is deliberate, curated
substring matching against a fixed clinical vocabulary (biomarkers,
receptor status, disease stage, treatment classes). Patient and trial term
sets are ranked against each other using the **Sørensen-Dice Index**
(`2·|X∩Y| / (|X|+|Y|)`), giving every trial a genuine rank position rather
than a present/absent flag.

### Rule-Based Hard-Exclusion Guardrail
A deterministic gate — not a model, not the cross-encoder — checks
pregnancy exclusion, age range, and keyword-matched exclusion criteria
(cardiac, hepatic, renal, autoimmune). Any trial that fails is dropped
before cross-encoder scoring runs. This is a binary safety gate, never a
weighted score component.

### Cross-Encoder Deep Scoring
A `cross-encoder/ms-marco-MiniLM-L-6-v2` model, fine-tuned on patient-trial
pairs (patient-level train/test split to prevent leakage), reads the full
patient summary and trial inclusion criteria together as one input pair —
capturing relationships a keyword or embedding-only approach misses.

### LLM-Powered Explanations & Chat (Groq — Llama 3.3 / Llama 4 Scout)
The LLM is used **only** for post-match explanation and a RAG-grounded
chat interface — never for the matching or ranking decision itself. This
is a deliberate design choice: the eligibility computation stays
deterministic and auditable, and LLM output is confined to where
hallucination risk doesn't translate into a patient-safety risk.

---

## Known Limitations (stated plainly, not hidden)

- **No physician- or outcome-validated ground truth.** Evaluation scripts
  (`evaluate_retrieval.py`) currently measure agreement with a
  self-generated rule-based proxy, not clinician judgment.
- **The keyword/entity vocabulary is a fixed, curated list**, not a
  learned model — it will miss phrasing outside that list, and patients
  with sparse structured data (e.g. missing receptor status) will have
  weak signal from this strategy specifically.
- **RRF's `k=60` constant is a standard default**, not tuned against this
  project's data.
- **The LLM chat/explanation layer is currently unevaluated** for
  faithfulness or groundedness — no automated check yet confirms its
  output only states facts present in the retrieved patient/trial text.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Vector Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| Keyword/Entity Overlap | Curated term list + Sørensen-Dice Index |
| Rank Fusion | Reciprocal Rank Fusion (RRF), k=60 |
| Cross-Encoder | Fine-tuned cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Rule Engine | Deterministic hard-exclusion guardrail |
| LLM | Groq API (Llama 3.3 70B / Llama 4 Scout) |
| Data Source | ClinicalTrials.gov (public), 19 real case-report patient profiles |

---

## Project Structure

```
clinical-trial-navigator/
├── app.py                  # Streamlit UI
├── config.py               # Constants, model names
├── pipeline/
│   ├── parser.py           # Patient profile extraction, trial text prep
│   ├── entity_search.py    # Keyword/entity overlap + Sørensen-Dice ranking
│   ├── embeddings.py       # Sentence transformer + ChromaDB operations
│   ├── scorer.py           # Rule-based guardrail + cross-encoder scoring
│   ├── rrf.py               # Reciprocal Rank Fusion (weight-free)
│   ├── matcher.py           # Full pipeline orchestration (Strategy A/B -> RRF -> CE -> RRF)
│   └── narrative.py        # Groq LLM explanations + RAG chat
├── data/
│   ├── patients/           # Real case-report-derived patient profiles
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
