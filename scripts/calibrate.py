"""Fit a temperature-scaling parameter for the cross-subject model.

Raw neural-net confidences are typically over-confident: a window scored 0.99 is not
really a seizure 99% of the time. Temperature scaling learns a single scalar T and
reports  P = sigmoid(logit / T)  instead of  sigmoid(logit), which rescales the whole
confidence curve so the numbers mean what they say. It changes calibration only, never
the ranking of windows.

The fitted T is written to a sidecar file next to the checkpoint
(models/calibration.json); src.predict picks it up automatically.

Honesty note: the deployable model was trained on all available patients, so this fits
T on data the model has seen — an optimistic calibration. For a rigorous estimate,
fit T on the leave-one-subject-out held-out predictions instead. The mechanism and
wiring are identical.

Run:  python -m scripts.calibrate            # default patients chb01 chb02 chb03
      python -m scripts.calibrate chb01      # a specific subset
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_checkpoint
from src.parse_summary import parse_summary
from src.cross_subject import load_and_window_multi

CKPT = "models/seizure_cnn2_final.pt"


def collect_logits(model, ckpt, patients):
    """Return (logits, labels) over all summarized recordings of the given patients."""
    channels = ckpt["channels"]
    window_sec = ckpt.get("window_sec", 5)
    all_logits, all_labels = [], []
    for pid in patients:
        summary = f"data/raw/{pid}/{pid}-summary.txt"
        if not os.path.exists(summary):
            print(f"  [skip] no summary for {pid}")
            continue
        seiz_by_file = parse_summary(summary)
        for edf in sorted(glob.glob(f"data/raw/{pid}/*.edf")):
            name = os.path.basename(edf)
            seizures = seiz_by_file.get(name, [])
            try:
                windows, labels = load_and_window_multi(edf, seizures, channels, window_sec)
            except Exception as e:
                print(f"  [skip] {name}: {e}")
                continue
            if len(windows) == 0:
                continue
            X = (windows - ckpt["norm_mean"]) / ckpt["norm_std"]
            X = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                logits = []
                for i in range(0, len(X), 256):
                    logits.append(model(X[i:i + 256]).squeeze(1))
                logits = torch.cat(logits).numpy()
            all_logits.append(logits)
            all_labels.append(labels)
            print(f"  {name}: {len(labels)} windows, {int(labels.sum())} seizure")
    if not all_logits:
        raise SystemExit("No data collected — check data/raw/<patient>/ and summaries.")
    return np.concatenate(all_logits), np.concatenate(all_labels)


def fit_temperature(logits, labels):
    """Fit T minimizing binary cross-entropy of sigmoid(logit / T)."""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) > 0
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / torch.exp(log_T), y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_T).item())
    with torch.no_grad():
        before = bce(z, y).item()
        after = bce(z / T, y).item()
    return T, before, after


def main():
    patients = sys.argv[1:] or ["chb01", "chb02", "chb03"]
    print(f"Calibrating on: {', '.join(patients)}")
    model, ckpt = load_checkpoint(CKPT)
    logits, labels = collect_logits(model, ckpt, patients)
    T, before, after = fit_temperature(logits, labels)

    out = os.path.join(os.path.dirname(CKPT), "calibration.json")
    with open(out, "w") as f:
        json.dump({
            "temperature": round(T, 4),
            "fit_on": patients,
            "n_windows": int(len(labels)),
            "n_seizure_windows": int(labels.sum()),
            "bce_before": round(before, 4),
            "bce_after": round(after, 4),
        }, f, indent=2)

    print(f"\nFitted temperature T = {T:.4f}")
    print(f"BCE  before={before:.4f}  after={after:.4f}")
    print(f"Saved → {out}  (src.predict applies it automatically)")


if __name__ == "__main__":
    main()