"""External validation: evaluate a CHB-MIT-trained seizure model on Siena.

The honest cross-dataset test — the model never saw Siena patients OR the Siena
montage in training. We harmonize Siena onto the model's exact bipolar montage
(src/siena.py), apply the model's own stored normalization (identical preprocessing
to training), and report window-level detection metrics overall and per subject.

Memory-safe: Siena recordings are long (some multi-GB EDFs expand to >6 GB in RAM).
We therefore score ONE FILE AT A TIME, keep only the tiny prob/label arrays, and
skip any single EDF larger than --max-mb (default 900) so a whale can't OOM the box.

Run:
  python -m src.eval_external
  python -m src.eval_external --model models/seizure_cnn2_final.pt --threshold 0.5 --max-mb 900
"""
import argparse
import gc
import os

import numpy as np
import torch
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

from src.predict import load_checkpoint, _score, normalize_windows
from src.risk_assessment import (smooth, windows_to_episodes, merge_and_filter_episodes,
                                 SMOOTH_K, MERGE_GAP_SEC, MIN_EPISODE_SEC)
from src.siena import (parse_siena_seizures, load_and_window_siena, _resolve_edf,
                       available_siena_subjects, DATA_ROOT)

WINDOW_SEC = 5


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _pipeline_events(prob, lab, thr):
    """Apply the LIVE app's post-processing (smooth → threshold → merge gaps → drop
    short episodes) to one recording, then score at the EVENT level:
      * a true seizure span is DETECTED if any surviving predicted episode overlaps it.
      * a predicted episode overlapping no true seizure is a FALSE ALARM.
    Returns (n_true_events, n_detected, n_false_alarms)."""
    preds = smooth(prob, SMOOTH_K) > thr
    pred_eps = merge_and_filter_episodes(
        windows_to_episodes(preds, WINDOW_SEC), MERGE_GAP_SEC, MIN_EPISODE_SEC)
    true_eps = windows_to_episodes(lab.astype(bool), WINDOW_SEC)
    detected = sum(1 for te in true_eps if any(_overlaps(te, pe) for pe in pred_eps))
    fa = sum(1 for pe in pred_eps if not any(_overlaps(pe, te) for te in true_eps))
    return len(true_eps), detected, fa


def _event_stats(lab, pred):
    """Seizure-level (event) stats within one recording (temporal order):
      * a labeled seizure EVENT (contiguous run of lab==1) counts as DETECTED if any
        window inside it is predicted positive.
      * a FALSE-ALARM event = a contiguous run of pred==1 that overlaps no true seizure.
    Returns (n_seizure_events, n_detected, n_false_alarm_events)."""
    n = len(lab)
    seiz = det = 0
    i = 0
    while i < n:
        if lab[i] == 1:
            j = i
            while j < n and lab[j] == 1:
                j += 1
            seiz += 1
            if pred[i:j].any():
                det += 1
            i = j
        else:
            i += 1
    fa = 0
    i = 0
    while i < n:
        if pred[i] == 1:
            j = i
            while j < n and pred[j] == 1:
                j += 1
            if not lab[i:j].any():
                fa += 1
            i = j
        else:
            i += 1
    return seiz, det, fa


def _metrics(y, prob, thr):
    pred = (prob > thr).astype(int)
    out = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "n": len(y), "pos": int(y.sum()), "pred_pos": int(pred.sum()),
    }
    try:
        out["auc"] = roc_auc_score(y, prob) if y.sum() and (y == 0).any() else float("nan")
    except ValueError:
        out["auc"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/seizure_cnn2_final.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-mb", type=float, default=900.0,
                    help="skip any single EDF larger than this (avoids OOM on multi-GB whales)")
    ap.add_argument("--subjects", default="",
                    help="comma list to restrict eval, e.g. 'PN05,PN14,PN17' (default: all)")
    args = ap.parse_args()

    subs = available_siena_subjects()
    if args.subjects:
        want = {s.strip() for s in args.subjects.split(",") if s.strip()}
        subs = [s for s in subs if s in want]
    if not subs:
        print("No Siena subjects under data/siena/. Download some first.")
        return
    print(f"Siena subjects available ({len(subs)}): {subs}")

    model, ckpt = load_checkpoint(args.model)
    channels = ckpt.get("channels")
    if not channels:
        print(f"{args.model} has no channel montage — need the cross-subject SeizureCNN2 model.")
        return
    print(f"Model: {ckpt.get('arch')} on {len(channels)} channels, trained on "
          f"{len(ckpt.get('subjects', []))} CHB-MIT subjects "
          f"(val F1 {ckpt.get('best_val_f1', float('nan')):.3f})")
    print(f"Scoring file-by-file (skip EDFs > {args.max_mb:.0f} MB)...\n")

    probs, ys, sids, skipped = [], [], [], []
    ev_seiz = ev_det = ev_fa = tot_win = 0        # raw event-level accumulators
    pl_true = pl_det = pl_fa = 0                   # clinical-pipeline (smoothed) accumulators
    for subj in subs:
        d = os.path.join(DATA_ROOT, subj)
        seiz = parse_siena_seizures(os.path.join(d, f"Seizures-list-{subj}.txt"))
        for fname, intervals in seiz.items():
            path = _resolve_edf(fname, d)
            if path is None:
                continue
            mb = os.path.getsize(path) / 1e6
            if mb > args.max_mb:
                skipped.append((subj, os.path.basename(path), mb))
                print(f"  ⚠ skip {subj}/{os.path.basename(path)} ({mb:.0f} MB > {args.max_mb:.0f})")
                continue
            try:
                w, lab, miss = load_and_window_siena(path, intervals, channels)
            except Exception as e:
                print(f"  !! skip {subj}/{os.path.basename(path)}: {type(e).__name__}: {e}")
                continue
            Xn = normalize_windows(w, ckpt)
            p = _score(model, torch.tensor(Xn, dtype=torch.float32), ckpt.get("temperature", 1.0))
            probs.append(p); ys.append(lab); sids.append(np.full(len(lab), subj))
            se, det_, fa = _event_stats(lab, (p > args.threshold).astype(int))
            ev_seiz += se; ev_det += det_; ev_fa += fa; tot_win += len(lab)
            pt, pd_, pf = _pipeline_events(p, lab, args.threshold)
            pl_true += pt; pl_det += pd_; pl_fa += pf
            print(f"  {subj}/{os.path.basename(path)}: {len(lab)} windows, {int(lab.sum())} seizure"
                  + (f", {miss} ch zero-filled" if miss else ""))
            del w, Xn; gc.collect()

    if not probs:
        print("\nNothing scored (all files skipped?). Try raising --max-mb.")
        return
    prob = np.concatenate(probs); y = np.concatenate(ys); sid = np.concatenate(sids)
    print(f"\nSiena test set: {len(y)} windows, {int(y.sum())} seizure "
          f"({100*y.mean():.1f}% positive)")

    thr = args.threshold
    print(f"\n=== Per-subject (threshold {thr}) ===")
    print(f"{'subj':6} {'windows':>8} {'seiz':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'auc':>6}")
    for s in sorted(set(sid)):
        m = sid == s
        r = _metrics(y[m], prob[m], thr)
        print(f"{s:6} {r['n']:>8} {r['pos']:>6} {r['precision']:>6.2f} "
              f"{r['recall']:>6.2f} {r['f1']:>6.2f} {r['auc']:>6.2f}")

    r = _metrics(y, prob, thr)
    tn, fp, fn, tp = confusion_matrix(y, (prob > thr).astype(int), labels=[0, 1]).ravel()
    print(f"\n=== OVERALL (threshold {thr}) ===")
    print(f"  precision {r['precision']:.3f}   recall {r['recall']:.3f}   "
          f"F1 {r['f1']:.3f}   AUC {r['auc']:.3f}")
    print(f"  confusion: TP {tp}  FP {fp}  FN {fn}  TN {tn}")

    hours = tot_win * 5 / 3600.0
    sens = ev_det / ev_seiz if ev_seiz else float("nan")
    print(f"\n=== EVENT-LEVEL, raw windows (threshold {thr}) ===")
    print(f"  seizure events: {ev_seiz}   detected: {ev_det}   sensitivity {sens:.1%}")
    print(f"  false-alarm events: {ev_fa} over {hours:.1f} h  ->  {ev_fa / hours:.2f} per hour"
          if hours else "  (no duration)")

    pl_sens = pl_det / pl_true if pl_true else float("nan")
    print(f"\n=== EVENT-LEVEL, LIVE PIPELINE (smooth k={SMOOTH_K} + merge {MERGE_GAP_SEC}s "
          f"+ min-episode {MIN_EPISODE_SEC}s, threshold {thr}) ===")
    print(f"  seizure events: {pl_true}   detected: {pl_det}   sensitivity {pl_sens:.1%}")
    print(f"  false-alarm episodes: {pl_fa} over {hours:.1f} h  ->  {pl_fa / hours:.2f} per hour"
          if hours else "  (no duration)")

    print(f"\n=== Window-level threshold sweep (overall) ===")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        m = _metrics(y, prob, t)
        print(f"  thr {t:.1f}:  F1 {m['f1']:.3f}   rec {m['recall']:.3f}   prec {m['precision']:.3f}")

    # The clinically meaningful operating curve: seizure sensitivity vs false alarms/hr,
    # AFTER the live smoothing/min-episode pipeline, across thresholds.
    print(f"\n=== CLINICAL-PIPELINE operating points (sensitivity vs false-alarms/hr) ===")
    print(f"  {'thr':>4} {'sensitivity':>12} {'detected':>10} {'FA/hr':>8}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        T = D = F = 0
        for pf, lf in zip(probs, ys):
            pt, pd_, pf_ = _pipeline_events(pf, lf, t)
            T += pt; D += pd_; F += pf_
        s = D / T if T else float("nan")
        print(f"  {t:>4.1f} {s:>11.1%} {f'{D}/{T}':>10} {F / hours:>8.2f}")

    if skipped:
        print(f"\nSkipped {len(skipped)} whale EDF(s) to protect memory:")
        for s, f, mb in skipped:
            print(f"  {s}/{f}  ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
