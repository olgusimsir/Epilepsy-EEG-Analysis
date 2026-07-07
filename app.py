"""
EEG Epilepsy Analysis — Streamlit app.

Pick or upload a CHB-MIT EDF recording. The app windows it, runs the chosen CNN,
shows a recording-level epilepsy risk assessment + timeline, and generates an
LLM-written clinical report (local Foundry Local, optional RAG grounding).

Run with:  streamlit run app.py
"""
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.parse_summary import parse_summary
from src.risk_assessment import assess_recording
from src.clinical_report import build_report
from src.predict import load_checkpoint, predict_recording

WINDOW_SEC = 5
OVERLAP = 0.5     # 50% overlapping windows at inference (matches serve_ui.py)
SAMPLE_DIR = "data/raw/chb01"
SUMMARY_PATH = "data/raw/chb01/chb01-summary.txt"

MODELS = {
    "Single-patient (chb01) — optimistic": "models/seizure_cnn.pt",
    "Cross-subject (7 patients) — honest": "models/seizure_cnn2_final.pt",
}
RISK_STYLE = {  # level -> streamlit status fn (box color already conveys severity)
    "High": st.error,
    "Moderate": st.warning,
    "Low": st.success,
}


@st.cache_resource
def load_model(checkpoint_path):
    """Load a checkpoint (model + metadata) once, cached across reruns."""
    return load_checkpoint(checkpoint_path)


# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EEG Epilepsy Analysis", page_icon="🧠", layout="wide")
st.title("EEG Epilepsy Analysis & Risk Assessment")
st.caption(
    "Detect seizure activity in scalp EEG, assess epilepsy risk, and generate a "
    "clinical report with a fully-local LLM. Research/educational — not a medical device."
)

# ── Sidebar: model ───────────────────────────────────────────────────────────
st.sidebar.header("① Detection model")
available = {name: path for name, path in MODELS.items() if os.path.exists(path)}
if not available:
    st.error("No trained model found. Run `python -m src.train` (single-patient) "
             "or `python -m src.train_final` (cross-subject) first.")
    st.stop()
model_choice = st.sidebar.selectbox("Model", list(available.keys()), label_visibility="collapsed")
model, ckpt = load_model(available[model_choice])
if "honest" in model_choice:
    st.sidebar.caption("Cross-patient model: ~0.4 F1 on unseen patients (realistic). "
                       "Expect more false positives than the single-patient demo.")

# ── Sidebar: input ───────────────────────────────────────────────────────────
st.sidebar.header("② Recording")
mode = st.sidebar.radio("Source", ["Sample (chb01)", "Upload your own"], label_visibility="collapsed")
threshold = st.sidebar.slider("Detection threshold", 0.0, 1.0, 0.5, 0.05)

edf_path = chosen_name = None
if mode == "Sample (chb01)":
    if os.path.isdir(SAMPLE_DIR):
        samples = sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".edf"))
        chosen_name = st.sidebar.selectbox("File", samples)
        edf_path = os.path.join(SAMPLE_DIR, chosen_name) if chosen_name else None
    else:
        st.sidebar.warning(f"Sample folder `{SAMPLE_DIR}` not found.")
else:
    uploaded = st.sidebar.file_uploader("EDF file", type=["edf"])
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
        tmp.write(uploaded.read())
        tmp.close()
        edf_path, chosen_name = tmp.name, uploaded.name

# ── Sidebar: report settings ─────────────────────────────────────────────────
st.sidebar.header("③ Report (LLM)")
provider_label = st.sidebar.selectbox(
    "Provider",
    ["Local — Foundry Local (offline)", "Cloud — Anthropic (sends data out)", "Template (no LLM)"],
)
use_rag = st.sidebar.checkbox("Ground with knowledge base (RAG)", value=True)

report_provider, report_model, report_key = "template", None, None
if provider_label.startswith("Local"):
    report_provider = "foundry_local"
    report_model = st.sidebar.text_input("Local model", value="phi-3.5-mini")
elif provider_label.startswith("Cloud"):
    report_provider = "anthropic"
    report_key = st.sidebar.text_input("Anthropic API key", type="password",
                                       help="Used only for this request; never stored.")

# ── Main: run ────────────────────────────────────────────────────────────────
if edf_path is None:
    st.info("← Pick a sample file or upload an EDF recording in the sidebar to begin.")
    st.stop()

with st.spinner(f"Analyzing {chosen_name}…"):
    probs, _preds = predict_recording(model, ckpt, edf_path, threshold=threshold,
                                      overlap=OVERLAP)
times = np.arange(len(probs)) * WINDOW_SEC
assessment = assess_recording(probs, window_sec=WINDOW_SEC, threshold=threshold)
# Display the same post-processed (smoothed/merged/filtered) episodes the score uses.
intervals = [(ep["start_sec"], ep["end_sec"]) for ep in assessment["episodes"]]

# ── Risk hero ────────────────────────────────────────────────────────────────
status_fn = RISK_STYLE[assessment["risk_level"]]
status_fn(f"**Epilepsy risk: {assessment['risk_level']}**  —  "
          f"score **{assessment['risk_score']}/100**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk score", f"{assessment['risk_score']}/100")
c2.metric("Abnormal windows", f"{assessment['n_abnormal_windows']} / {assessment['n_windows']}")
c3.metric("Episodes", assessment["n_episodes"])
c4.metric("Peak confidence", f"{assessment['peak_confidence']:.2f}")

# ── Detection details ────────────────────────────────────────────────────────
left, right = st.columns([3, 2])
with left:
    if intervals:
        st.markdown("**Detected episodes**")
        for s, e in intervals:
            st.markdown(f"- `{s}s – {e}s`  ({e - s}s)")
    else:
        st.markdown("**No seizure-like activity detected.**")
with right:
    true_intervals = []
    if chosen_name and os.path.exists(SUMMARY_PATH):
        true_intervals = parse_summary(SUMMARY_PATH).get(chosen_name, [])
    if true_intervals:
        st.markdown("**Ground truth (dataset)**")
        for s, e in true_intervals:
            st.markdown(f"- `{s}s – {e}s`")

# ── Timeline plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 2.8))
ax.plot(times, probs, color="#2563eb", linewidth=1.1, label="Seizure probability")
ax.axhline(threshold, color="#9ca3af", linestyle="--", linewidth=1, label=f"Threshold ({threshold:.2f})")
for s, e in intervals:
    ax.axvspan(s, e, color="#ef4444", alpha=0.22)
for s, e in true_intervals:
    ax.axvspan(s, e, color="#16a34a", alpha=0.0, hatch="///", edgecolor="#16a34a", linewidth=0.0)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("P(seizure)")
ax.set_ylim(0, 1)
ax.set_title("Per-window seizure probability   (red = detected, green hatch = ground truth)")
ax.legend(loc="upper right", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig)

# ── Expert report ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Clinical EEG report")
rag_note = " · grounded with the local knowledge base (RAG)" if use_rag else ""
st.caption(f"Generated by **{provider_label}**{rag_note}. Decision-support draft only.")

if st.button("Generate report", type="primary"):
    with st.spinner("Generating report… (first local run loads the model)"):
        report_text, source = build_report(
            assessment, chosen_name or "uploaded.edf",
            provider=report_provider, api_key=report_key or None,
            model=report_model, use_rag=use_rag,
        )
    st.caption(f"Source: `{source}`")
    st.markdown(report_text)

st.divider()
st.caption(
    "Research/educational demo only — not a medical device. Trained on CHB-MIT scalp "
    "EEG; predictions and reports are decision-support, not a diagnosis. With the local "
    "provider, no patient data leaves this machine."
)