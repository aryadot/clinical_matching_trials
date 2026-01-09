

#  Clinical Trial Matching System (Breast Cancer)

## Overview

This project is an end-to-end **clinical trial discovery and recommendation system** that helps surface the most relevant clinical trials for a patient based on their clinical profile.

The system takes **synthetic patient summaries** and **public clinical trial data** and ranks trials using interpretable clinical signals, presenting the **top matches as percentages** in a simple, clinician-style web interface.

This project is intended for **educational and research purposes only** and does **not** make medical or eligibility determinations.

---

## Problem Motivation

Finding relevant clinical trials is difficult for patients and time-consuming for clinicians due to:

* long and complex eligibility criteria
* unstructured clinical text
* lack of personalized trial discovery tools

This project reframes trial discovery as a **relevance-ranking problem**, rather than strict eligibility enforcement, making it easier to explore promising trials while maintaining transparency.

---

## Key Features

*  **Patient-specific trial ranking**
*  **Match percentage scoring** (relative relevance)
*  **Interpretable clinical signals**
*  Uses **synthetic patient data** (no real PHI)
*  **Streamlit web app** for interactive exploration
*  Modular pipeline (preprocessing → scoring → UI)

---

## So How It Works

### 1. Data Sources

* **Clinical trial data** from ClinicalTrials.gov (public)
* **Synthetic patient summaries** derived from published case reports

No real patient data is used.

---

### 2. Trial Scoring Logic

Each patient–trial pair is scored using multiple interpretable signals:

| Signal            | Description                   |
| ----------------- | ----------------------------- |
| Disease relevance | Breast cancer focus alignment |
| Malignancy        | Cancer vs benign condition    |
| Biomarkers        | HER2 / ER mentions            |
| Disease stage     | Metastatic relevance          |
| Demographics      | Age availability              |
| Safety checks     | Pregnancy exclusions          |

These signals are combined into a **raw relevance score**.

---

### 3. Match Percentage

To make results intuitive:

* Raw scores are **normalized per patient**
* Scores are converted into **percentages**
* The **top 5 trials** are displayed for each patient

> A higher percentage indicates stronger relative relevance for that specific patient.

---

### 4. Web Application

The Streamlit app allows users to:

* select a patient profile
* view the top 5 matching clinical trials
* see **why** each trial was recommended
* explore results in a clean, clinician-style interface


## Running the Project Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate trial rankings

```bash
python matching/match_trials.py
```

### 3. Launch the web app

```bash
streamlit run app.py
```




