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
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
    .stApp { background: linear-gradient(180deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%); }

    .main-header { text-align: center; padding: 1.5rem 0 1rem 0; }
    .main-header h1 {
        font-family: 'DM Sans', sans-serif; font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #22d3ee 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .glass-panel {
        background: rgba(22, 27, 34, 0.6); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 1.2rem; backdrop-filter: blur(10px); margin-bottom: 0.8rem;
    }
    .glass-panel h3 {
        font-family: 'DM Sans', sans-serif; color: #e6edf3;
        font-size: 1rem; font-weight: 600; margin-bottom: 0.8rem;
    }

    .trial-card {
        background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
        transition: border-color 0.2s;
    }
    .trial-card:hover { border-color: rgba(99,102,241,0.3); }
    .trial-title { color: #e6edf3; font-size: 0.9rem; font-weight: 600; line-height: 1.4; }
    .trial-meta { color: #8b949e; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; margin-top: 0.3rem; }

    .match-badge {
        font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700;
        padding: 4px 12px; border-radius: 8px; display: inline-block;
    }
    .match-high { background: rgba(46,160,67,0.15); color: #2ea043; }
    .match-med { background: rgba(210,153,34,0.15); color: #d29922; }
    .match-low { background: rgba(139,148,158,0.15); color: #8b949e; }

    .score-bar { height: 6px; border-radius: 3px; margin-top: 0.3rem; }
    .score-bar-fill { height: 100%; border-radius: 3px; }

    .entity-tag {
        display: inline-block; font-size: 0.68rem; font-weight: 600;
        padding: 2px 8px; border-radius: 4px; margin: 2px 2px;
    }
    .entity-biomarker { background: rgba(99,102,241,0.15); color: #a78bfa; }
    .entity-cancer { background: rgba(248,81,73,0.15); color: #f85149; }
    .entity-treatment { background: rgba(46,160,67,0.15); color: #2ea043; }
    .entity-stage { background: rgba(210,153,34,0.15); color: #d29922; }

    .reason-item { font-size: 0.8rem; color: rgba(230,237,243,0.7); padding: 0.2rem 0; }

    .stat-card {
        background: rgba(22,27,34,0.8); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 1rem; text-align: center;
    }
    .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #6366f1; }
    .stat-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }

    .disclaimer {
        text-align: center; color: rgba(139,148,158,0.35); font-size: 0.72rem;
        padding: 2rem 0 1rem 0; max-width: 650px; margin: 0 auto;
    }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
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
        <h1>🧬 Clinical Trial Navigator</h1>
        <p style="color: #8b949e; font-size: 0.95rem; max-width: 600px; margin: 0.3rem auto; line-height: 1.5;">
            AI-powered clinical trial discovery using <span style="color: #6366f1;">semantic search</span>,
            <span style="color: #22d3ee;">clinical NER</span>, and
            <span style="color: #a78bfa;">LLM-powered explanations</span>.
            Select a patient profile to find the most relevant trials.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading data..."):
        patients, trials, trial_texts = load_data()

    # ── Sidebar: Patient Selection ──
    st.sidebar.markdown("### 👤 Select Patient")
    patient_options = {p["patient_id"]: p for p in patients}
    selected_id = st.sidebar.selectbox(
        "Patient",
        list(patient_options.keys()),
        format_func=lambda x: f"Patient {x} (Age: {patient_options[x]['age']})",
    )
    patient = patient_options[selected_id]

    # Show patient summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Patient Profile")
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
    if st.button("🔍 Find Matching Trials", type="primary", use_container_width=True):
        run_matching(patient, trials, trial_texts)
    elif "last_results" in st.session_state and st.session_state.get("last_patient") == selected_id:
        display_results(patient, st.session_state.last_results)
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0; max-width: 500px; margin: 0 auto;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.15;">🧬</div>
            <p style="color: #8b949e; line-height: 1.6; font-size: 0.9rem;">
                Select a patient from the sidebar and click <strong>Find Matching Trials</strong>
                to run the three-layer NLP matching pipeline.
            </p>
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
        st.markdown('<div class="glass-panel"><h3>🏥 Top Matching Trials</h3></div>', unsafe_allow_html=True)

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
                        <div class="trial-meta" style="margin-top: 0.2rem;">💊 {safe_interventions}</div>
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
        <div class="glass-panel">
            <h3>💬 Ask About Matches</h3>
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
        This tool is for informational and educational purposes only. It does not constitute medical advice
        or clinical trial eligibility determination. Always consult qualified healthcare professionals.
        <br><br>
        Built with Streamlit · Sentence Transformers · ChromaDB · spaCy · Groq Llama 3.3
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
