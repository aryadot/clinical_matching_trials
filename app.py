"""
Clinical Trial Navigator — AI-Powered Trial Discovery
======================================================
Matches patients to clinical trials using NLP: semantic embeddings,
clinical NER, rule-based signals, and LLM-powered explanations.
"""

import streamlit as st
import json
from html import escape
from pathlib import Path
from config import PATIENTS_RAW, TRIALS_PARSED

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Clinical Trial Navigator",
    page_icon="CT",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    /* ── Healthcare Light Theme ── */
    .stApp {
        background: #f0f4f8 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #334155 !important;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 2px solid #0d9488;
        margin-bottom: 1.5rem;
        background: linear-gradient(180deg, #ffffff 0%, #f0fdfa 100%);
        border-radius: 0 0 20px 20px;
        padding-bottom: 1.5rem;
    }
    .main-header h1 {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #0f766e;
        margin-bottom: 0.2rem;
    }
    .main-header .subtitle {
        color: #64748b;
        font-size: 0.92rem;
        max-width: 620px;
        margin: 0.4rem auto;
        line-height: 1.55;
        font-family: 'Source Sans 3', sans-serif;
    }
    .main-header .subtitle span.hl-teal { color: #0d9488; font-weight: 600; }
    .main-header .subtitle span.hl-blue { color: #2563eb; font-weight: 600; }
    .main-header .subtitle span.hl-purple { color: #7c3aed; font-weight: 600; }

    /* Card panels — white with subtle shadows */
    .clinical-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 0.8rem;
    }
    .clinical-panel h3 {
        font-family: 'Source Sans 3', sans-serif;
        color: #0f172a;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ccfbf1;
    }

    /* Trial result cards */
    .trial-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0d9488;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: all 0.2s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .trial-card:hover {
        border-left-color: #0f766e;
        box-shadow: 0 4px 12px rgba(13,148,136,0.1);
        transform: translateY(-1px);
    }
    .trial-title {
        color: #0f172a;
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        line-height: 1.4;
    }
    .trial-meta {
        color: #64748b;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 0.3rem;
    }

    /* Match percentage badges */
    .match-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 8px;
        display: inline-block;
    }
    .match-high { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .match-med { background: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }
    .match-low { background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; }

    /* Entity tags — clinical color coding */
    .entity-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 2px 2px;
    }
    .entity-biomarker { background: #ede9fe; color: #6d28d9; border: 1px solid #ddd6fe; }
    .entity-cancer { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }
    .entity-treatment { background: #d1fae5; color: #059669; border: 1px solid #a7f3d0; }
    .entity-stage { background: #fef3c7; color: #d97706; border: 1px solid #fde68a; }

    .reason-item {
        font-size: 0.8rem;
        color: #475569;
        padding: 0.2rem 0;
        font-family: 'Source Sans 3', sans-serif;
    }

    /* Stat cards — teal accent */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 3px solid #0d9488;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stat-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f766e;
    }
    .stat-label {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }

    /* Score breakdown labels */
    .score-label {
        font-size: 0.7rem;
        color: #94a3b8;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Disclaimer */
    .disclaimer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.72rem;
        padding: 2rem 0 1rem 0;
        max-width: 650px;
        margin: 0 auto;
        border-top: 1px solid #e2e8f0;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 0;
        max-width: 500px;
        margin: 0 auto;
    }
    .empty-state .icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .empty-state h2 {
        font-family: 'Source Sans 3', sans-serif;
        color: #0f172a;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .empty-state p {
        color: #64748b;
        line-height: 1.6;
        font-size: 0.9rem;
    }

    /* Pipeline feature cards */
    .pipeline-cards {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.8rem;
        margin-top: 1.5rem;
    }
    .pipeline-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .pipeline-card .icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
    .pipeline-card h4 {
        font-family: 'Source Sans 3', sans-serif;
        color: #0f172a; font-size: 0.85rem; font-weight: 600; margin: 0.3rem 0;
    }
    .pipeline-card p { color: #64748b; font-size: 0.72rem; line-height: 1.4; margin: 0; }

    /* CSS Icons — replacing emojis with professional shapes */
    .icon-dna {
        width: 28px; height: 28px; border-radius: 8px;
        background: linear-gradient(135deg, #0d9488, #2dd4bf);
        display: inline-flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .icon-dna::after { content: "DNA"; }

    .icon-search-med {
        width: 24px; height: 24px; border: 2px solid #0d9488;
        border-radius: 50%; display: inline-block; position: relative;
    }
    .icon-search-med::after {
        content: ""; position: absolute; width: 8px; height: 2px;
        background: #0d9488; bottom: -2px; right: -4px;
        transform: rotate(45deg);
    }

    .icon-dot {
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block; margin-right: 6px;
    }
    .icon-dot-teal { background: #0d9488; }
    .icon-dot-blue { background: #2563eb; }
    .icon-dot-purple { background: #7c3aed; }
    .icon-dot-green { background: #059669; }
    .icon-dot-amber { background: #d97706; }

    .pipeline-icon {
        width: 40px; height: 40px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 0.5rem auto;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700; font-size: 0.7rem; color: white;
    }
    .pipeline-icon-embed { background: linear-gradient(135deg, #0d9488, #14b8a6); }
    .pipeline-icon-embed::after { content: "VEC"; }
    .pipeline-icon-ner { background: linear-gradient(135deg, #2563eb, #60a5fa); }
    .pipeline-icon-ner::after { content: "NER"; }
    .pipeline-icon-llm { background: linear-gradient(135deg, #7c3aed, #a78bfa); }
    .pipeline-icon-llm::after { content: "LLM"; }

    .section-icon {
        width: 24px; height: 24px; border-radius: 6px;
        display: inline-flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700; font-size: 0.55rem; color: white;
        margin-right: 6px; vertical-align: middle;
    }
    .section-icon-trials { background: #0d9488; }
    .section-icon-trials::after { content: "Rx"; }
    .section-icon-chat { background: #2563eb; }
    .section-icon-chat::after { content: "AI"; }
    .section-icon-patient { background: #7c3aed; }
    .section-icon-patient::after { content: "Pt"; }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Override Streamlit button */
    .stButton > button[kind="primary"] {
        background: #0d9488 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #0f766e !important;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data
def load_data():
    from pipeline.parser import load_patients, load_trials, build_trial_embedding_text
    patients = load_patients(PATIENTS_RAW)
    trials = load_trials(TRIALS_PARSED)
    trial_texts = [build_trial_embedding_text(t) for t in trials]
    return patients, trials, trial_texts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1><span class="icon-dna"></span> Clinical Trial Navigator</h1>
        <p class="subtitle">
            Intelligent trial discovery for oncology patients. Our NLP pipeline uses
            <span class="hl-teal">semantic embeddings</span> to find relevant trials,
            <span class="hl-blue">clinical NER</span> to extract biomarkers and conditions, and
            <span class="hl-purple">LLM-powered reasoning</span> to explain each match.
            Select a patient profile to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading data..."):
        patients, trials, trial_texts = load_data()

    # ── Sidebar: Patient Selection ──
    st.sidebar.markdown("### Select Patient")
    patient_options = {p["patient_id"]: p for p in patients}
    selected_id = st.sidebar.selectbox(
        "Patient",
        list(patient_options.keys()),
        format_func=lambda x: f"Patient {x} (Age: {patient_options[x]['age']})",
    )
    patient = patient_options[selected_id]

    # Show patient summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Patient Profile")
    st.sidebar.write(patient["raw_summary"][:500])

    rs = patient["receptor_status"]
    tags = ""
    for receptor, val in rs.items():
        if val != "unknown":
            tags += f'<span class="entity-tag entity-biomarker">{receptor}: {val}</span>'
    if patient["metastatic"]:
        tags += '<span class="entity-tag entity-stage">Metastatic</span>'
    if patient.get("stage", "unknown") != "unknown":
        tags += f'<span class="entity-tag entity-stage">{patient["stage"]}</span>'
    if patient["pregnant"]:
        tags += '<span class="entity-tag entity-cancer">Pregnant</span>'
    if tags:
        st.sidebar.markdown(tags, unsafe_allow_html=True)

    # ── Run Matching Pipeline ──
    if st.button("Find Matching Trials", type="primary", use_container_width=True):
        run_matching(patient, trials, trial_texts)
    elif "last_results" in st.session_state and st.session_state.get("last_patient") == selected_id:
        display_results(patient, st.session_state.last_results)
    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div style="width: 56px; height: 56px; border-radius: 14px; background: linear-gradient(135deg, #0d9488, #2dd4bf); margin: 0 auto 1rem auto; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.9rem;">CT</span>
            </div>
            <h2>Ready to Match Patients to Trials</h2>
            <p>
                Select a patient from the sidebar and click <strong>Find Matching Trials</strong>
                to run the three-layer NLP matching pipeline across 1,046 breast cancer clinical trials.
            </p>
        </div>
        <div class="pipeline-cards" style="max-width: 700px; margin: 0 auto;">
            <div class="pipeline-card">
                <div class="pipeline-icon pipeline-icon-embed"></div>
                <h4>Semantic Search</h4>
                <p>Sentence-transformer embeddings find trials with similar clinical profiles via ChromaDB vector search</p>
            </div>
            <div class="pipeline-card">
                <div class="pipeline-icon pipeline-icon-ner"></div>
                <h4>Clinical NER</h4>
                <p>spaCy extracts biomarkers (HER2, ER, BRCA), cancer types, treatments, and disease stages</p>
            </div>
            <div class="pipeline-card">
                <div class="pipeline-icon pipeline-icon-llm"></div>
                <h4>LLM Explanations</h4>
                <p>Groq generates clinician-friendly match explanations and powers follow-up Q&A</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def run_matching(patient, trials, trial_texts):
    """Execute the full matching pipeline."""
    from pipeline.parser import build_patient_embedding_text
    from pipeline.embeddings import index_trials, semantic_search
    from pipeline.ner import extract_clinical_entities
    from pipeline.scorer import score_patient_trial, rank_trials

    progress = st.progress(0, "Indexing trials into vector store...")

    # Step 1: Index trials into ChromaDB
    collection = index_trials(trials, trial_texts)
    progress.progress(25, "Generating patient embedding...")

    # Step 2: Semantic search
    patient_text = build_patient_embedding_text(patient)
    semantic_results = semantic_search(patient_text, collection, top_k=30)
    progress.progress(50, "Extracting clinical entities...")

    # Step 3: NER on patient
    patient_entities = extract_clinical_entities(patient["raw_summary"])

    # Step 4: Score each semantic match with all three signals
    trial_lookup = {t["trial_id"]: t for t in trials}
    scored = []

    for i, match in enumerate(semantic_results):
        trial = trial_lookup.get(match["trial_id"], {})
        trial_text = f"{trial.get('conditions', '')} {trial.get('interventions', '')} {trial.get('title', '')}"
        trial_entities = extract_clinical_entities(trial_text)

        result = score_patient_trial(
            patient=patient,
            trial=trial,
            semantic_similarity=match["similarity"],
            patient_entities=patient_entities,
            trial_entities=trial_entities,
        )
        scored.append(result)
        progress.progress(50 + int((i / len(semantic_results)) * 40), "Scoring trials...")

    # Step 5: Rank
    top_matches = rank_trials(scored, top_k=10)
    progress.progress(100, "Done!")

    # Save to session state
    st.session_state.last_results = top_matches
    st.session_state.last_patient = patient["patient_id"]
    st.session_state.trial_chat_history = []

    display_results(patient, top_matches)


def display_results(patient, top_matches):
    """Display the ranked trial matches."""
    if not top_matches:
        st.warning("No matching trials found for this patient profile.")
        return

    # ── Stats Row ──
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{len(top_matches)}</div><div class="stat-label">Matches Found</div></div>', unsafe_allow_html=True)
    with s2:
        best = top_matches[0]["match_percentage"] if top_matches else 0
        st.markdown(f'<div class="stat-card"><div class="stat-value">{best}%</div><div class="stat-label">Best Match</div></div>', unsafe_allow_html=True)
    with s3:
        avg_sem = sum(t["semantic_score"] for t in top_matches) / len(top_matches) if top_matches else 0
        st.markdown(f'<div class="stat-card"><div class="stat-value">{avg_sem:.0%}</div><div class="stat-label">Avg Similarity</div></div>', unsafe_allow_html=True)
    with s4:
        avg_ner = sum(t["ner_score"] for t in top_matches) / len(top_matches) if top_matches else 0
        st.markdown(f'<div class="stat-card"><div class="stat-value">{avg_ner:.0%}</div><div class="stat-label">Entity Overlap</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Trial Cards ──
    results_col, chat_col = st.columns([2, 1])

    with results_col:
        st.markdown('<div class="clinical-panel"><h3><span class="section-icon section-icon-trials"></span> Top Matching Trials</h3></div>', unsafe_allow_html=True)

        for i, trial in enumerate(top_matches):
            pct = trial["match_percentage"]
            badge_class = "match-high" if pct >= 70 else "match-med" if pct >= 40 else "match-low"
            bar_color = "#2ea043" if pct >= 70 else "#d29922" if pct >= 40 else "#8b949e"

            safe_title = escape(trial.get("title", "Unknown Trial")[:120])
            safe_conditions = escape(trial.get("conditions", "N/A")[:100])
            safe_interventions = escape(trial.get("interventions", "N/A")[:100])

            reasons_html = "".join(
                f'<div class="reason-item">• {escape(r)}</div>'
                for r in trial.get("reasons", [])[:4]
            )

            # Score breakdown
            sem_pct = int(trial["semantic_score"] * 100)
            rule_pct = int(trial["rule_score"] * 100)
            ner_pct = int(trial["ner_score"] * 100)

            st.markdown(f"""
            <div class="trial-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 1;">
                        <div class="trial-title">{i+1}. {safe_title}</div>
                        <div class="trial-meta">{trial['trial_id']} · {safe_conditions}</div>
                        <div class="trial-meta" style="margin-top: 0.2rem;"><span class="icon-dot icon-dot-green"></span>{safe_interventions}</div>
                    </div>
                    <div class="match-badge {badge_class}">{pct}%</div>
                </div>
                <div style="margin-top: 0.6rem; display: flex; gap: 1rem; font-size: 0.7rem; color: #8b949e;">
                    <span>Semantic: {sem_pct}%</span>
                    <span>Rules: {rule_pct}%</span>
                    <span>NER: {ner_pct}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.05); height: 4px; border-radius: 2px; margin-top: 0.4rem;">
                    <div style="width: {pct}%; height: 100%; background: {bar_color}; border-radius: 2px;"></div>
                </div>
                <div style="margin-top: 0.5rem;">{reasons_html}</div>
            </div>
            """, unsafe_allow_html=True)

    with chat_col:
        st.markdown("""
        <div class="clinical-panel">
            <h3><span class="section-icon section-icon-chat"></span> Ask About Matches</h3>
            <p style="color: #8b949e; font-size: 0.8rem;">
                Ask follow-up questions about this patient's trial matches.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize chat
        if "trial_chat_history" not in st.session_state:
            st.session_state.trial_chat_history = []

        for msg in st.session_state.trial_chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("e.g. 'Why is trial #1 the best match?'"):
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                from pipeline.narrative import chat_with_context
                with st.spinner("Thinking..."):
                    response = chat_with_context(prompt, patient, top_matches)
                st.write(response)

            st.session_state.trial_chat_history.append({"role": "user", "content": prompt})
            st.session_state.trial_chat_history.append({"role": "assistant", "content": response})

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This tool is for <strong>informational and educational purposes only</strong>. It does not constitute medical advice,
        clinical trial eligibility determination, or treatment recommendation. Always consult qualified healthcare professionals.
        <br><br>
        Built with Streamlit · Sentence Transformers · ChromaDB · spaCy · Groq Llama 3.3
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
