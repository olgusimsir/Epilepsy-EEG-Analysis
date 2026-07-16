# NeuroScan — EEG Epilepsy Seizure Detection & Risk Assessment

An end-to-end, **fully-local** decision-support tool for scalp EEG. It detects
seizure activity with a convolutional neural network, turns per-window predictions
into a recording-level epilepsy-risk assessment, and drafts a clinical report with a
local language model — all on your own machine, with no patient data leaving it.

📹 **Demo video:** https://drive.google.com/file/d/11zgckpcWk4Vsc9Sf-ix3OLxdcKytCCVt/view?usp=sharing

> **Not a medical device.** This is a research / decision-support prototype, not a
> diagnostic tool. Every finding requires review by a qualified neurologist.

---

## What it does

Upload a scalp-EEG recording (`.edf`) → the tool:

1. **Detects seizures** — a 1-D CNN scores every 5-second window for seizure activity.
2. **Assesses risk** — smooths the window scores, groups them into episodes, and
   produces a 0–100 risk score with a High / Moderate / Low level.
3. **Writes a report** — a local LLM drafts a clinical narrative from the analysis,
   grounded in reference literature (RAG). Falls back to a deterministic template if
   no LLM is configured.

Served as a web app (`serve_ui.py`) at `http://localhost:8000`.

## The two "AIs" in this project

| Component | Role | Where |
|---|---|---|
| **`SeizureCNN2`** (the core AI) | Reads raw EEG → seizure probability per window. This is the trained deep-learning model doing the detection. | `src/model.py`, `src/train_final.py`, `src/predict.py` |
| **Local LLM** (secondary) | Writes the report *prose* from the CNN's findings. It never sees raw EEG. | `src/clinical_report.py`, `src/llm/` |

---

## Results

### In-dataset (CHB-MIT)
Trained on **all 24 CHB-MIT patients**. The deployed model `seizure_cnn2_final.pt`
reaches **val F1 ≈ 0.72**; the per-window-normalized variant `seizure_cnn2_zscore.pt`
reaches **val F1 ≈ 0.79**. On held-out CHB-MIT seizure files it reliably localizes the
true seizure (e.g. `chb01_03`: detected 3000–3030 s vs. ground-truth 2996–3036 s).

### Cross-dataset generalization study (the honest part)
Trained on CHB-MIT, tested **zero-shot on the Siena Scalp EEG database** — different
patients *and* different hardware/montage (referential vs. bipolar, 512 vs. 256 Hz),
harmonized onto the model's montage (`src/siena.py`). 8 Siena subjects, ~22k windows:

| Setup | AUC | Seizure sensitivity (event-level) | False alarms / hr |
|---|---|---|---|
| Global-scalar normalization | 0.68 | 44 % (8/18) | 13.4 |
| **Per-window normalization** | **0.76** | **67 % (12/18)** | 36.8 |
| Per-window + live pipeline (smoothing + min-episode) | 0.76 | 56 % (10/18) | **7.6** |

Key findings — all measured, not assumed:
- **Per-window z-score normalization** substantially improves cross-dataset transfer
  (AUC 0.68 → 0.76, sensitivity 44 % → 67 %).
- The **live post-processing pipeline** (smoothing + minimum-episode duration) cuts
  false alarms ~5× (37 → 7.6/hr) for a modest sensitivity cost.
- A threshold sweep reveals a usable operating point: at threshold 0.9, **0.8 false
  alarms/hr at 44 % sensitivity** — within the clinical <1–2/hr bar.
- **Fine-tuning** on a few Siena subjects did *not* help (a reported negative result):
  too little data → overfitting, more false alarms.

**Bottom line:** the model carries real, transferable signal, but like all research
seizure detectors it is **not clinically deployable** on unseen data. The value here
is a rigorous, honest cross-dataset evaluation — not an inflated accuracy number.

---

## How the pipeline works

```
 .edf  ─►  window (5 s, 23 bipolar ch, 256 Hz)  ─►  SeizureCNN2  ─►  P(seizure) per window
                                                                           │
                                       smooth (k=3) → threshold → merge/min-episode
                                                                           │
                                        risk score (0–100) + level + episodes
                                                                           │
                                          local LLM (+ RAG)  ─►  clinical report
```

- **Model** (`SeizureCNN2`): Conv1d → BatchNorm → ReLU → MaxPool (×2), a third conv,
  global average pooling, dropout, linear classifier. Built to generalize across
  patients (BatchNorm + GAP + dropout rather than a huge flatten layer).
- **Montage-agnostic input**: accepts CHB-MIT bipolar EDFs directly, and derives the
  bipolar montage from referential electrodes (Siena-style) on the fly, resampling to
  256 Hz (`src/predict._montage_data`).
- **Risk assessment** (`src/risk_assessment.py`): moving-average smoothing, episode
  merging (gaps ≤ 10 s) and a 10-second minimum episode length remove single-window
  false spikes.

---

## Project layout

```
serve_ui.py             Web app (stdlib HTTP server) → http://localhost:8000
design/                 Frontend (vanilla HTML/CSS/JS + a decorative Three.js brain)
src/
  model.py              SeizureCNN / SeizureCNN2 architectures
  predict.py            Inference: montage handling, normalization, windowing, scoring
  risk_assessment.py    Per-window probs → episodes → 0–100 risk score
  train_final.py        Train the deployable model (--norm global | per_window)
  train_cross_subject.py  Leave-one-subject-out generalization test
  cross_subject.py      Multi-subject dataset builder (common montage, per-subject caps)
  siena.py              Siena → CHB-MIT montage harmonization (external test set)
  eval_external.py      Evaluate any model on Siena (window + event-level metrics)
  finetune_siena.py     Domain-adaptation experiment (train on some Siena subjects)
  subjects.py           Auto-discover downloaded patients
  clinical_report.py    Orchestrates the LLM / template report
  llm/                  Providers: foundry_local (default), anthropic (opt-in), template
  rag/                  Retrieval-augmented grounding for the report
scripts/
  download_chbmit.py    Fetch CHB-MIT patients from PhysioNet
  download_siena.py     Fetch the Siena external test set
  download_foundry_model.py   Download + warm a local LLM (Foundry Local)
  calibrate.py          Temperature-scale a model's probabilities
models/                 Trained checkpoints (gitignored; rebuild by training)
tests/                  Unit tests (risk assessment)
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the web tool

```bash
# with the local LLM report (recommended):
PATH="/opt/homebrew/bin:$PATH" PORT=8000 WARMUP_LLM=1 python serve_ui.py
# then open http://localhost:8000
```

A plain `python serve_ui.py` also works — the **analysis** runs regardless; only the
LLM report needs Foundry Local (otherwise it uses the offline template).

## Get the data

```bash
# CHB-MIT (training + in-dataset test) — seizure files only keeps it compact
python -m scripts.download_chbmit --subjects chb01-chb24 --seizure-only

# Siena (external test set)
python -m scripts.download_siena
python -m src.subjects        # verify what was downloaded
```

Data goes under `data/raw/chbNN/` and `data/siena/PNxx/` (both gitignored).

## Train & evaluate

```bash
# Train the deployable model (global normalization) on every downloaded patient
python -m src.train_final

# Train the cross-dataset-robust variant (per-window normalization)
python -m src.train_final --norm per_window --out models/seizure_cnn2_zscore.pt

# Honest within-CHB-MIT generalization (leave-one-subject-out)
python -m src.train_cross_subject

# External validation on Siena (window + event-level metrics + operating curve)
python -m src.eval_external --model models/seizure_cnn2_zscore.pt
```

## Local LLM report (Foundry Local)

The report is generated on-device — no data leaves the machine.

```bash
# one-time: install the Foundry Local runtime + a model
brew tap microsoft/foundrylocal && brew install foundrylocal
python scripts/download_foundry_model.py phi-3.5-mini   # or qwen2.5-0.5b (small/fast)
```

Providers (`src/llm/factory.py`): `foundry_local` (default, local), `anthropic`
(opt-in, cloud — sends the analysis off-device), `template` (deterministic, offline).

> Note: if `venv/` is copied between folders, the Foundry SDK's ONNX-runtime symlinks
> can go stale — recreate the venv from `requirements.txt` rather than copying it.

## Docker

```bash
docker build -t neuroscan .
docker run --rm -p 8000:8000 neuroscan   # analysis works; LLM report needs extra wiring
```

---

## Datasets & credits

- **CHB-MIT Scalp EEG Database** — PhysioNet (`physionet.org/content/chbmit/`).
- **Siena Scalp EEG Database** — PhysioNet (`physionet.org/content/siena-scalp-eeg/`).
- Decorative 3D brain model: Science Museum Group (CC BY-SA 4.0).

## Disclaimer

NeuroScan is a research prototype for **decision support**, not a medical device and
not a diagnosis. Its cross-dataset evaluation confirms it is not reliable enough for
clinical use on unseen recordings. A qualified neurologist must confirm every finding.
