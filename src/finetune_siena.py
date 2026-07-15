"""Domain-adapt the per-window CHB-MIT model to Siena by fine-tuning on a SUBSET of
Siena patients, holding the rest out for an honest test.

Train subjects: PN00, PN06, PN12, PN13, PN16.  Held-out test (never seen here):
PN05, PN14, PN17 — evaluate afterwards with:
    python -m src.eval_external --model models/seizure_cnn2_finetuned.pt --subjects PN05,PN14,PN17
and compare to the zero-shot model on the SAME subjects:
    python -m src.eval_external --model models/seizure_cnn2_zscore.pt   --subjects PN05,PN14,PN17

Memory-safe: loads one EDF at a time, caps normals per subject, skips >900 MB whales.
Uses per-window z-score (matches the base model's norm_mode).
"""
import copy
import gc
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from src.predict import load_checkpoint
from src.train import evaluate
from src.siena import parse_siena_seizures, load_and_window_siena, _resolve_edf, DATA_ROOT

TRAIN_SUBJECTS = ["PN00", "PN06", "PN12", "PN13", "PN16"]   # test held out: PN05, PN14, PN17
BASE_MODEL = "models/seizure_cnn2_zscore.pt"
OUT_PATH = "models/seizure_cnn2_finetuned.pt"
MAX_NORMALS_PER_SUBJECT = 600
MAX_MB = 900


def _zscore(X):
    m = X.mean(axis=2, keepdims=True)
    s = X.std(axis=2, keepdims=True) + 1e-8
    return ((X - m) / s).astype(np.float32)


def build_train(channels, seed=42):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for subj in TRAIN_SUBJECTS:
        d = os.path.join(DATA_ROOT, subj)
        seiz = parse_siena_seizures(os.path.join(d, f"Seizures-list-{subj}.txt"))
        sX, sY = [], []
        for fname, iv in seiz.items():
            p = _resolve_edf(fname, d)
            if p is None or os.path.getsize(p) / 1e6 > MAX_MB:
                continue
            try:
                w, lab, _ = load_and_window_siena(p, iv, channels)
            except Exception as e:
                print(f"  skip {subj}/{os.path.basename(p)}: {e}")
                continue
            sX.append(w); sY.append(lab)
            print(f"  {subj}/{os.path.basename(p)}: {len(lab)} win, {int(lab.sum())} seizure")
            del w; gc.collect()
        if not sX:
            continue
        sx = np.concatenate(sX); sy = np.concatenate(sY)
        pos = np.where(sy == 1)[0]; neg = np.where(sy == 0)[0]
        if len(neg) > MAX_NORMALS_PER_SUBJECT:
            neg = rng.choice(neg, MAX_NORMALS_PER_SUBJECT, replace=False)
        keep = np.sort(np.concatenate([pos, neg]))
        X_list.append(_zscore(sx[keep])); y_list.append(sy[keep])
        del sx, sy; gc.collect()
    X = np.concatenate(X_list).astype(np.float32)
    y = np.concatenate(y_list)
    return X, y


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = load_checkpoint(BASE_MODEL)
    channels = ckpt["channels"]
    print(f"Fine-tuning {BASE_MODEL} on Siena {TRAIN_SUBJECTS}  (device {device})\n")

    X, y = build_train(channels)
    print(f"\ntrain windows: {len(y)}, seizure {int(y.sum())} ({100 * y.mean():.1f}% positive)")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = model.to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

    nneg, npos = (yt == 0).sum().item(), (yt == 1).sum().item()
    w = torch.tensor([1.0 / nneg, 1.0 / npos])[yt.long().squeeze(1)]
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64,
                        sampler=WeightedRandomSampler(w, len(w), replacement=True))

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)   # low LR — gentle adaptation
    best_f1, best_state, noimp = -1.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        f1 = evaluate(model, Xv, yv)["f1"]
        if f1 > best_f1:
            best_f1, best_state, noimp = f1, copy.deepcopy(model.state_dict()), 0
        else:
            noimp += 1
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1}: val F1 {f1:.3f} (best {best_f1:.3f})")
        if noimp >= 8:
            print(f"early stop at epoch {epoch+1}"); break

    model.load_state_dict(best_state)
    out = {k: ckpt[k] for k in ["arch", "n_channels", "channels", "norm_mode",
                                "norm_mean", "norm_std", "window_sec"]}
    out.update({"model_state": best_state, "subjects": ckpt.get("subjects"),
                "finetuned_on": TRAIN_SUBJECTS, "best_val_f1": best_f1})
    torch.save(out, OUT_PATH)
    print(f"\nSaved fine-tuned model to {OUT_PATH}  (val F1 {best_f1:.3f})")


if __name__ == "__main__":
    main()
