"""
Clinical Trial Navigator — AI-Powered Trial Discovery
======================================================
Matches patients to clinical trials using NLP: semantic embeddings,
a fine-tuned BERT cross-encoder, rule-based hard filters, and LLM-powered explanations.
"""

import os
import streamlit as st
import json
from html import escape
from pathlib import Path
from config import PATIENTS_RAW, TRIALS_PARSED

st.set_page_config(
    page_title="Clinical Trial Navigator",
    page_icon="CT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    .stApp { background: #f0f4f8 !important; }
    section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }
    section[data-testid="stSidebar"] * { color: #334155 !important; }
    .main-header { text-align: center; padding: 2rem 0 1.5rem 0; border-bottom: 2px solid #0d9488; margin-bottom: 1.5rem; background: linear-gradient(180deg, #ffffff 0%, #f0fdfa 100%); border-radius: 0 0 20px 20px; }
    .main-header h1 { font-family: 'Source Sans 3', sans-serif; font-size: 2.4rem; font-weight: 700; color: #0f766e; margin-bottom: 0.2rem; }
    .main-header .subtitle { color: #64748b; font-size: 0.92rem; max-width: 620px; margin: 0.4rem auto; line-height: 1.55; font-family: 'Source Sans 3', sans-serif; }
    .main-header .subtitle span.hl-teal { color: #0d9488; font-weight: 600; }
    .main-header .subtitle span.hl-blue { color: #2563eb; font-weight: 600; }
    .main-header .subtitle span.hl-purple { color: #7c3aed; font-weight: 600; }
    .clinical-panel { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 0.8rem; }
    .clinical-panel h3 { font-family: 'Source Sans 3', sans-serif; color: #0f172a; font-size: 1rem; font-weight: 700; margin-bottom: 0.8rem; padding-bottom: 0.5rem; border-bottom: 2px solid #ccfbf1; }
    .trial-card { background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #0d9488; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.6rem; transition: all 0.2s; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .trial-card:hover { border-left-color: #0f766e; box-shadow: 0 4px 12px rgba(13,148,136,0.1); transform: translateY(-1px); }
    .trial-title { color: #0f172a; font-family: 'Source Sans 3', sans-serif; font-size: 0.9rem; font-weight: 600; line-height: 1.4; }
    .trial-meta { color: #64748b; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; margin-top: 0.3rem; }
    .match-badge { font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; padding: 4px 12px; border-radius: 8px; display: inline-block; }
    .match-high { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .match-med { background: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }
    .match-low { background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; }
    .entity-tag { display: inline-block; font-size: 0.68rem; font-weight: 600; padding: 3px 10px; border-radius: 12px; margin: 2px 2px; }
    .entity-biomarker { background: #ede9fe; color: #6d28d9; border: 1px solid #ddd6fe; }
    .entity-cancer { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }
    .entity-treatment { background: #d1fae5; color: #059669; border: 1px solid #a7f3d0; }
    .entity-stage { background: #fef3c7; color: #d97706; border: 1px solid #fde68a; }
    .reason-item { font-size: 0.8rem; color: #475569; padding: 0.2rem 0; font-family: 'Source Sans 3', sans-serif; }
    .stat-card { background: #ffffff; border: 1px solid #e2e8f0; border-top: 3px solid #0d9488; border-radius: 10px; padding: 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #0f766e; }
    .stat-label { font-family: 'Source Sans 3', sans-serif; font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
    .disclaimer { text-align: center; color: #94a3b8; font-size: 0.72rem; padding: 2rem 0 1rem 0; max-width: 650px; margin: 0 auto; border-top: 1px solid #e2e8f0; }
    .empty-state { text-align: center; padding: 4rem 0; max-width: 500px; margin: 0 auto; }
    .empty-state h2 { font-family: 'Source Sans 3', sans-serif; color: #0f172a; font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; }
    .empty-state p { color: #64748b; line-height: 1.6; font-size: 0.9rem; }
    .pipeline-cards { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem; margin-top: 1.5rem; }
    .pipeline-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .pipeline-card h4 { font-family: 'Source Sans 3', sans-serif; color: #0f172a; font-size: 0.85rem; font-weight: 600; margin: 0.3rem 0; }
    .pipeline-card p { color: #64748b; font-size: 0.72rem; line-height: 1.4; margin: 0; }
    .icon-dna { width: 28px; height: 28px; border-radius: 8px; background: linear-gradient(135deg, #0d9488, #2dd4bf); display: inline-flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; }
    .icon-dna::after { content: "DNA"; }
    .icon-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
    .icon-dot-green { background: #059669; }
    .pipeline-icon { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.5rem auto; font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.7rem; color: white; }
    .pipeline-icon-embed { background: linear-gradient(135deg, #0d9488, #14b8a6); }
    .pipeline-icon-embed::after { content: "VEC"; }
    .pipeline-icon-ce { background: linear-gradient(135deg, #2563eb, #60a5fa); }
    .pipeline-icon-ce::after { content: "CE"; }
    .pipeline-icon-llm { background: linear-gradient(135deg, #7c3aed, #a78bfa); }
    .pipeline-icon-llm::after { content: "LLM"; }
    .section-icon { width: 24px; height: 24px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.55rem; color: white; margin-right: 6px; vertical-align: middle; }
    .section-icon-trials { background: #0d9488; }
    .section-icon-trials::after { content: "Rx"; }
    .section-icon-chat { background: #2563eb; }
    .section-icon-chat::after { content: "AI"; }
    .section-icon-patient { background: #7c3aed; }
    .section-icon-patient::after { content: "Pt"; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton > button[kind="primary"] { background: #0d9488 !important; border: none !important; border-radius: 10px !important; font-family: 'Source Sans 3', sans-serif !important; font-weight: 600 !important; }
    .stButton > button[kind="primary"]:hover { background: #0f766e !important; }

    /* Chat widgets (st.chat_input / st.chat_message) default to dark-theme
       text colors, which are invisible on this app's light background —
       force readable colors explicitly. */
    [data-testid="stChatInput"] textarea { color: #0f172a !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #94a3b8 !important; opacity: 1 !important; }
    [data-testid="stChatInput"] { background: #ffffff !important; border: 1px solid #e2e8f0 !important; }
    [data-testid="stChatMessage"] { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 10px !important; }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] div { color: #1e293b !important; }
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
# Helper Functions — defined BEFORE main()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _render_empty_state():
    st.markdown("""
    <div class="empty-state">
        <div style="width: 56px; height: 56px; border-radius: 14px; background: linear-gradient(135deg, #0d9488, #2dd4bf); margin: 0 auto 1rem auto; display: flex; align-items: center; justify-content: center;">
            <span style="color: white; font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.9rem;">CT</span>
        </div>
        <h2>Ready to Match Patients to Trials</h2>
        <p>Select a patient from the sidebar and click <strong>Find Matching Trials</strong> to run the NLP matching pipeline across 1,046 breast cancer clinical trials.</p>
    </div>
    <div class="pipeline-cards" style="max-width: 700px; margin: 0 auto;">
        <div class="pipeline-card"><div class="pipeline-icon pipeline-icon-embed"></div><h4>Semantic Search</h4><p>Sentence-transformer embeddings find trials with similar clinical profiles via ChromaDB</p></div>
        <div class="pipeline-card"><div class="pipeline-icon pipeline-icon-ce"></div><h4>Cross-Encoder Scoring</h4><p>BERT cross-encoder scores patient-trial pairs semantically — no string matching needed</p></div>
        <div class="pipeline-card"><div class="pipeline-icon pipeline-icon-llm"></div><h4>Criterion Checker</h4><p>LLM reviews each inclusion criterion against your profile and flags what to discuss with your oncologist</p></div>
    </div>
    """, unsafe_allow_html=True)


def render_patient_form():
    from pipeline.eligibility import build_patient_from_form
    st.markdown("""
    <div class="clinical-panel">
        <h3><span class="section-icon section-icon-patient"></span> Enter Your Clinical Profile</h3>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0;">Fill in your clinical details below. This information is used only to find potentially relevant trials — it is never stored or shared.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        diagnosis = st.text_input("Primary Diagnosis", placeholder="e.g. Invasive ductal carcinoma, metastatic breast cancer")
        stage = st.selectbox("Disease Stage", ["unknown", "early", "locally advanced", "metastatic"],
                             format_func=lambda x: x.capitalize() if x != "unknown" else "Unknown / Not sure")
    with col2:
        her2 = st.selectbox("HER2 Status", ["unknown", "positive", "negative"], format_func=lambda x: x.capitalize())
        er   = st.selectbox("ER Status",   ["unknown", "positive", "negative"], format_func=lambda x: x.capitalize())
        pr   = st.selectbox("PR Status",   ["unknown", "positive", "negative"], format_func=lambda x: x.capitalize())

    prior_treatments = st.multiselect("Prior Treatments Received", [
        "AC-T chemotherapy", "Taxane-based chemotherapy", "Anthracycline chemotherapy",
        "Trastuzumab (Herceptin)", "Pertuzumab", "T-DM1 (Kadcyla)",
        "Trastuzumab deruxtecan (Enhertu)", "CDK4/6 inhibitor (palbociclib/ribociclib/abemaciclib)",
        "Endocrine therapy (tamoxifen/aromatase inhibitor)", "Immunotherapy (checkpoint inhibitor)",
        "PARP inhibitor (olaparib/niraparib)", "Carboplatin", "Capecitabine",
        "Radiation therapy", "Surgery (lumpectomy)", "Surgery (mastectomy)"
    ], placeholder="Select all that apply")

    comorbidities = st.multiselect("Relevant Comorbidities", [
        "Pregnancy", "Cardiac disease", "Renal disease", "Hepatic disease",
        "Autoimmune condition", "Diabetes", "Active infection", "Brain metastases"
    ], placeholder="Select all that apply")

    additional_notes = st.text_area("Additional Clinical Notes (optional)", placeholder="Any other relevant clinical information...", height=80)

    st.markdown("""
    <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 0.8rem 1rem; margin: 1rem 0; font-size: 0.8rem; color: #92400e;">
        <strong>Important:</strong> This tool identifies trials that may be worth discussing with your oncologist. It does not determine clinical eligibility or constitute medical advice.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Find Matching Trials", type="primary", use_container_width=True):
        if not diagnosis and her2 == "unknown" and er == "unknown" and pr == "unknown":
            st.error("Please fill in at least your diagnosis or receptor status.")
            return None
        return build_patient_from_form(
            age=age, diagnosis=diagnosis, her2=her2, er=er, pr=pr,
            stage=stage, prior_treatments=prior_treatments,
            comorbidities=[c.lower() for c in comorbidities],
            additional_notes=additional_notes
        )
    return None


def display_eligibility_check(patient: dict, trial: dict):
    from pipeline.eligibility import check_eligibility
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not groq_api_key:
        st.warning("GROQ_API_KEY not configured.")
        return
    with st.spinner("Checking eligibility criteria..."):
        result = check_eligibility(patient, trial, groq_api_key)
    if not result.get("criteria_assessment"):
        st.info(result.get("overall_summary", "No eligibility criteria available."))
        return
    st.markdown(f"""
    <div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.8rem; font-size: 0.85rem; color: #166534;">
        <strong>Summary:</strong> {result['overall_summary']}
    </div>""", unsafe_allow_html=True)
    st.markdown("**Inclusion Criteria Review:**")
    for item in result["criteria_assessment"]:
        status    = item.get("status", "UNCLEAR")
        note      = item.get("note", "")
        criterion = item.get("criterion", "")
        if status == "MEETS":
            color, icon, bg, border = "#166534", "✓", "#f0fdf4", "#86efac"
        elif status == "DOES NOT MEET":
            color, icon, bg, border = "#dc2626", "✗", "#fef2f2", "#fecaca"
        else:
            color, icon, bg, border = "#d97706", "?", "#fffbeb", "#fde68a"
        st.markdown(f"""
        <div style="background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 0.5rem 0.8rem; margin-bottom: 0.3rem; font-size: 0.78rem;">
            <span style="color: {color}; font-weight: 700;">{icon} {status}</span>
            <span style="color: #374151; margin-left: 0.5rem;">{criterion[:100]}</span>
            {f'<div style="color: #6b7280; margin-top: 0.2rem; font-size: 0.72rem;">↳ {note}</div>' if note else ''}
        </div>""", unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.7rem; color: #9ca3af; margin-top: 0.5rem;">⚕️ Discuss with your oncologist before taking any action.</div>', unsafe_allow_html=True)


def run_matching(patient, trials, trial_texts, show_eligibility=False):
    from pipeline.embeddings import index_trials
    from pipeline.entity_search import build_trial_entity_index
    from pipeline.matcher import match_patient_to_trials

    progress = st.progress(0, "Indexing trials into vector store...")
    collection = index_trials(trials, trial_texts)

    @st.cache_resource(show_spinner=False)
    def _cached_entity_index(_trials_key):
        return build_trial_entity_index(trials)

    progress.progress(10, "Building keyword/entity index...")
    trial_entity_index = _cached_entity_index(len(trials))

    def _on_progress(pct, msg):
        progress.progress(pct, msg)

    top_matches = match_patient_to_trials(
        patient=patient, trials=trials, collection=collection,
        trial_entity_index=trial_entity_index, top_k=10,
        on_progress=_on_progress,
    )
    st.session_state.last_results = top_matches
    st.session_state.last_patient = patient["patient_id"]
    st.session_state.trial_chat_history = []
    display_results(patient, top_matches, show_eligibility=show_eligibility)


def display_results(patient, top_matches, show_eligibility=False):
    if not top_matches:
        st.warning("No matching trials found for this patient profile.")
        return

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{len(top_matches)}</div><div class="stat-label">Matches Found</div></div>', unsafe_allow_html=True)
    with s2:
        best = top_matches[0]["eligibility_pct"] if top_matches else 0
        st.markdown(f'<div class="stat-card"><div class="stat-value">{best}%</div><div class="stat-label">Best Match</div></div>', unsafe_allow_html=True)
    with s3:
        avg_sem = sum(t["semantic_score"] for t in top_matches) / len(top_matches) if top_matches else 0
        st.markdown(f'<div class="stat-card"><div class="stat-value">{avg_sem:.0%}</div><div class="stat-label">Avg Similarity</div></div>', unsafe_allow_html=True)
    with s4:
        avg_ce = sum(t["cross_encoder_score"] for t in top_matches) / len(top_matches) if top_matches else 0
        st.markdown(f'<div class="stat-card"><div class="stat-value">{avg_ce:.0%}</div><div class="stat-label">Avg CE Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    results_col, chat_col = st.columns([2, 1])

    with results_col:
        st.markdown('<div class="clinical-panel"><h3><span class="section-icon section-icon-trials"></span> Top Matching Trials</h3></div>', unsafe_allow_html=True)
        for i, trial in enumerate(top_matches):
            pct = trial["eligibility_pct"]
            badge_class = "match-high" if pct >= 70 else "match-med" if pct >= 40 else "match-low"
            bar_color   = "#2ea043" if pct >= 70 else "#d29922" if pct >= 40 else "#8b949e"
            safe_title        = escape(trial.get("title", "Unknown Trial")[:120])
            safe_conditions   = escape(trial.get("conditions", "N/A")[:100])
            safe_interventions = escape(trial.get("interventions", "N/A")[:100])
            reasons_html = "".join(f'<div class="reason-item">• {escape(r)}</div>' for r in trial.get("reasons", [])[:4])
            entities_html = ""
            if trial.get("matched_entities"):
                entities_html = f'<div class="reason-item">• Matched terms: {escape(", ".join(trial["matched_entities"][:6]))}</div>'
            sem_pct   = int(trial["semantic_score"] * 100)
            dice_pct  = int(trial["dice_score"] * 100)
            ce_pct    = int(trial["cross_encoder_score"] * 100)
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
                    <span>Semantic: {sem_pct}%</span><span>Keyword/Dice: {dice_pct}%</span><span>Cross-Encoder: {ce_pct}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.05); height: 4px; border-radius: 2px; margin-top: 0.4rem;">
                    <div style="width: {pct}%; height: 100%; background: {bar_color}; border-radius: 2px;"></div>
                </div>
                <div style="margin-top: 0.5rem;">{reasons_html}{entities_html}</div>
            </div>
            """, unsafe_allow_html=True)

            if show_eligibility:
                btn_key = f"elig_{trial['trial_id']}_{i}"
                if st.button(f"Check Eligibility Criteria for Trial {i+1}", key=btn_key):
                    from pipeline.parser import load_trials
                    from config import TRIALS_PARSED
                    full_trials = load_trials(TRIALS_PARSED)
                    full_trial_lookup = {t["trial_id"]: t for t in full_trials}
                    full_trial = full_trial_lookup.get(trial["trial_id"], trial)
                    display_eligibility_check(patient, full_trial)

    with chat_col:
        st.markdown("""
        <div class="clinical-panel">
            <h3><span class="section-icon section-icon-chat"></span> Ask About Matches</h3>
            <p style="color: #8b949e; font-size: 0.8rem;">Ask follow-up questions about this patient's trial matches.</p>
        </div>
        """, unsafe_allow_html=True)
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

    st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This tool is for <strong>informational and educational purposes only</strong>. It does not constitute medical advice, clinical trial eligibility determination, or treatment recommendation. Always consult qualified healthcare professionals.
        <br><br>Built with Streamlit · Sentence Transformers · ChromaDB · BERT Cross-Encoder · Groq Llama 4 Scout
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main App — called AFTER all functions defined
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    st.markdown("""
    <div class="main-header">
        <h1><span class="icon-dna"></span> Clinical Trial Navigator</h1>
        <p class="subtitle">
            Intelligent trial discovery for oncology patients. Our NLP pipeline uses
            <span class="hl-teal">semantic embeddings</span> to find relevant trials,
            <span class="hl-blue">BERT cross-encoder scoring</span> for semantic relevance, and
            <span class="hl-purple">criterion-level eligibility checking</span> to explain each match.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        patients, trials, trial_texts = load_data()

    tab1, tab2 = st.tabs(["Demo Patient Profiles", "Enter My Profile"])

    with tab1:
        st.sidebar.markdown("### Select Patient")
        patient_options = {p["patient_id"]: p for p in patients}
        selected_id = st.sidebar.selectbox(
            "Patient", list(patient_options.keys()),
            format_func=lambda x: f"Patient {x} (Age: {patient_options[x]['age']})",
        )
        patient = patient_options[selected_id]
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

        if st.button("Find Matching Trials", type="primary", use_container_width=True, key="demo_btn"):
            run_matching(patient, trials, trial_texts)
        elif "last_results" in st.session_state and st.session_state.get("last_patient") == selected_id:
            display_results(patient, st.session_state.last_results)
        else:
            _render_empty_state()

    with tab2:
        form_patient = render_patient_form()
        if form_patient:
            st.session_state.form_patient = form_patient
            run_matching(form_patient, trials, trial_texts, show_eligibility=True)
        elif "form_patient" in st.session_state and "last_results" in st.session_state:
            display_results(st.session_state.form_patient, st.session_state.last_results, show_eligibility=True)


if __name__ == "__main__":
    main()