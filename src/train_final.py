"""Train the FINAL deployable model: SeizureCNN2 on all available patients.

Unlike train_cross_subject.py (which holds each patient out to *measure*
generalization), this trains on everything we have to produce one shippable model.
The checkpoint stores the channel montage and normalization stats so inference can
reproduce the exact preprocessing on any new recording.

Run:  python -m src.train_final
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from src.cross_subject import common_channels, build_multi_dataset
from src.model import SeizureCNN2
from src.subjects import available_subjects
from src.train import evaluate

# Train the shippable model on every subject we have (including seizure-free ones,
# which contribute useful negatives) — not just a hardcoded list.
SUBJECTS = available_subjects(require_seizure=False)
WINDOW_SEC = 5
OUT_PATH = "models/seizure_cnn2_final.pt"
# Max NORMAL windows per subject kept during the build (all seizures always kept),
# so the full 24-subject set (~13 GB) never has to fit in RAM (machine has 9 GB).
MAX_NORMALS_PER_SUBJECT = 500


def _zscore_per_window(X):
    """Per-window, per-channel z-score (zero-mean/unit-var over time). Removes each
    recording's amplitude/DC offset → far better cross-dataset transfer."""
    m = X.mean(axis=2, keepdims=True)
    s = X.std(axis=2, keepdims=True) + 1e-8
    return ((X - m) / s).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--norm", choices=["global", "per_window"], default="global",
                    help="'global' = one dataset-wide mean/std (classic, deployed model); "
                         "'per_window' = z-score each window/channel (cross-dataset robust)")
    ap.add_argument("--out", default=OUT_PATH, help="checkpoint output path")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}  |  normalization: {args.norm}  |  out: {args.out}")

    channels = common_channels(SUBJECTS)
    print(f"Common montage: {len(channels)} channels")
    # Cap normals PER SUBJECT during the build (keep all seizures) so the full
    # ~13 GB set is never loaded into this 9 GB machine.
    X, y, sid = build_multi_dataset(
        SUBJECTS, channels, max_normals_per_subject=MAX_NORMALS_PER_SUBJECT
    )
    print(f"Total after per-subject normal cap: {X.shape[0]} windows, {int(y.sum())} seizure "
          f"(~{X.nbytes/1e9:.1f} GB)")

    # Hold out a small stratified validation set just for early stopping.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    if args.norm == "per_window":
        # each window/channel normalized independently — no dataset-wide stats needed
        mean, std = 0.0, 1.0
        X_tr = _zscore_per_window(X_tr)
        X_val_n = _zscore_per_window(X_val)
    else:
        mean, std = X_tr.mean(), X_tr.std()
        X_tr = (X_tr - mean) / std
        X_val_n = (X_val - mean) / std
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

    # Class-balanced sampling.
    nn_, ns_ = (y_tr_t == 0).sum().item(), (y_tr_t == 1).sum().item()
    w = torch.tensor([1.0 / nn_, 1.0 / ns_])[y_tr_t.long().squeeze(1)]
    sampler = WeightedRandomSampler(w, len(w), replacement=True)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, sampler=sampler)

    model = SeizureCNN2(n_channels=len(channels)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1, best_state, no_improve = -1.0, None, 0
    for epoch in range(80):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        f1 = evaluate(model, X_val_t, y_val_t)["f1"]
        if f1 > best_f1:
            best_f1, best_state, no_improve = f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}: val F1 {f1:.3f} (best {best_f1:.3f})")
        if no_improve >= 12:
            print(f"early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "arch": "SeizureCNN2",
            "model_state": best_state,
            "n_channels": len(channels),
            "channels": channels,
            "norm_mode": args.norm,
            "norm_mean": float(mean),
            "norm_std": float(std),
            "window_sec": WINDOW_SEC,
            "subjects": SUBJECTS,
            "best_val_f1": best_f1,
        },
        args.out,
    )
    print(f"\nSaved final model to {args.out}  (norm={args.norm}, best val F1 {best_f1:.3f})")


if __name__ == "__main__":
    main()