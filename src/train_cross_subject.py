"""
Cross-subject seizure detection via leave-one-subject-out (LOSO) cross-validation,
with an A/B comparison of two configurations:

  baseline  — global normalization + SeizureCNN (flatten -> big FC)
  improved  — per-channel z-norm    + SeizureCNN2 (BatchNorm + global pool + dropout)

For each patient the model trains on the OTHER patients and is tested on this
held-out patient, so the score measures true generalization to a new person.

Run:  python -m src.train_cross_subject
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from src.cross_subject import common_channels, build_multi_dataset, zscore_per_channel
from src.model import SeizureCNN, SeizureCNN2
from src.train import evaluate

SUBJECTS = ["chb01", "chb02", "chb03", "chb05", "chb06", "chb07", "chb08"]


def _train(model, X_tr, y_tr, X_val, y_val, device):
    """Train one model with class-balanced sampling, early-stopping on val F1."""
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    num_normal = (y_tr_t == 0).sum().item()
    num_seizure = (y_tr_t == 1).sum().item()
    class_weights = torch.tensor([1.0 / num_normal, 1.0 / num_seizure])
    sample_weights = class_weights[y_tr_t.long().squeeze(1)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    loader = DataLoader(TensorDataset(X_tr, y_tr_t), batch_size=64, sampler=sampler)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1, best_state, no_improve = -1.0, None, 0
    for epoch in range(60):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        f1 = evaluate(model, X_val, y_val_t)["f1"]
        if f1 > best_f1:
            best_f1, best_state, no_improve = f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if no_improve >= 10:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_loso(X, y, sid, n_channels, normalize, model_factory, device, label):
    """One full LOSO sweep for a given (normalization, model) configuration."""
    print(f"\n--- {label} ---")
    X_proc = zscore_per_channel(X) if normalize == "per_channel" else X

    agg = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    per_fold_f1 = []
    for held in SUBJECTS:
        test_mask = sid == held
        train_mask = ~test_mask
        if int(y[test_mask].sum()) == 0:
            continue

        X_tr, y_tr = X_proc[train_mask], y[train_mask]
        X_te, y_te = X_proc[test_mask], y[test_mask]
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr
        )

        # Global normalization is applied per-fold from training stats; per-channel
        # is already done up front (per-window, so no leakage).
        if normalize == "global":
            mean, std = X_tr2.mean(), X_tr2.std()
            X_tr2, X_val = (X_tr2 - mean) / std, (X_val - mean) / std
            X_te_eval = (X_te - mean) / std
        else:
            X_te_eval = X_te

        model = model_factory(n_channels).to(device)
        model = _train(model, X_tr2, y_tr2, X_val, y_val, device)

        X_te_t = torch.tensor(X_te_eval, dtype=torch.float32).to(device)
        y_te_t = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1).to(device)
        m = evaluate(model, X_te_t, y_te_t)
        per_fold_f1.append(m["f1"])
        for k in agg:
            agg[k] += m[k]
        print(f"  [test {held}] F1={m['f1']:.3f}  P={m['precision']:.2f}  R={m['recall']:.2f}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print(f"  => mean-fold F1={np.mean(per_fold_f1):.3f} | micro F1={f1:.3f}  P={p:.2f}  R={r:.2f}")
    return {"mean_f1": float(np.mean(per_fold_f1)), "micro_f1": f1, "precision": p, "recall": r}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    channels = common_channels(SUBJECTS)
    print(f"Common montage: {len(channels)} channels")
    X, y, sid = build_multi_dataset(SUBJECTS, channels)
    print(f"\nTotal: {X.shape[0]} windows, {int(y.sum())} seizure")

    # Cap normal windows to bound memory (we have ~30x more normals than seizures,
    # and the balanced sampler oversamples seizures anyway). Keep ALL seizures.
    rng = np.random.default_rng(42)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    max_neg = 8000
    if len(neg) > max_neg:
        neg = rng.choice(neg, max_neg, replace=False)
    keep = np.sort(np.concatenate([pos, neg]))
    X, y, sid = X[keep], y[keep], sid[keep]
    print(f"After capping normals: {X.shape[0]} windows, {int(y.sum())} seizure "
          f"(~{X.nbytes/1e9:.1f} GB)\n")

    # 2x2: {global, per-channel} normalization x {SeizureCNN, SeizureCNN2}
    configs = [
        ("global", lambda c: SeizureCNN(n_channels=c),  "baseline: global + CNN"),
        ("global", lambda c: SeizureCNN2(n_channels=c), "improved: global + CNN2"),
    ]
    results = []
    for norm, factory, label in configs:
        results.append((label, run_loso(X, y, sid, len(channels), norm, factory, device, label)))

    print("\n=== 2x2 summary (cross-subject, micro-averaged) ===")
    print(f"{'config':<28} {'F1':>6} {'Precision':>10} {'Recall':>8}")
    for label, r in results:
        print(f"{label:<28} {r['micro_f1']:>6.3f} {r['precision']:>10.2f} {r['recall']:>8.2f}")


if __name__ == "__main__":
    main()