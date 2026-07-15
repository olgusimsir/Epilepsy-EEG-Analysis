"""Siena Scalp EEG as an EXTERNAL test set for the CHB-MIT-trained seizure CNN.

Siena is referential/monopolar (electrodes Fp1, F3, T3, ...) at 512 Hz; the model
was trained on CHB-MIT's 23 *bipolar* channels at 256 Hz. This module harmonizes
Siena onto that exact montage so the model can be evaluated on it unchanged:

  * derive each CHB-MIT bipolar channel by subtracting two Siena electrodes
    (e.g. FP1-F7 = Fp1 - F7), applying the old->new nomenclature (T3->T7, T4->T8,
    T5->P7, T6->P8) and using F9/F10 as proxies for CHB-MIT's FT9/FT10 (Siena has
    no FT9/FT10). If a needed electrode is absent for a subject, that channel is
    zero-filled — the montage varies per patient.
  * resample 512 -> 256 Hz, window at 5 s, and label a window seizure if it
    overlaps any annotated seizure interval.

This module returns RAW windows (in volts), exactly like cross_subject.build_multi_dataset.
The evaluator (eval_external.py) then applies the deployed model's own stored global
normalization ((X - norm_mean) / norm_std from the checkpoint) so Siena is preprocessed
identically to how the model was trained — amplitude/scale mismatch between datasets is
the main cross-dataset risk this honest test is meant to expose.
"""
import glob
import os
import re

import mne
import numpy as np

WINDOW_SEC = 5
TARGET_SFREQ = 256
DATA_ROOT = "data/siena"
# A few Siena seizure end-times are typo'd past the recording end (e.g. PN00-3 says
# a 61-min seizure). Guard against those turning most of a file into "seizure": clamp
# to the EDF duration, then cap any remaining implausibly-long seizure to this length.
MAX_SEIZURE_SEC = 300

# CHB-MIT bipolar channel  ->  (Siena electrode +, Siena electrode -)
# Electrode names are normalized (no "EEG " prefix, uppercase). T7/T8/P7/P8 map to
# Siena's T3/T4/T5/T6; FT9/FT10 approximated by F9/F10.
BIPOLAR_MAP = {
    "C3-P3":   ("C3", "P3"),
    "C4-P4":   ("C4", "P4"),
    "CZ-PZ":   ("CZ", "PZ"),
    "F3-C3":   ("F3", "C3"),
    "F4-C4":   ("F4", "C4"),
    "F7-T7":   ("F7", "T3"),
    "F8-T8":   ("F8", "T4"),
    "FP1-F3":  ("FP1", "F3"),
    "FP1-F7":  ("FP1", "F7"),
    "FP2-F4":  ("FP2", "F4"),
    "FP2-F8":  ("FP2", "F8"),
    "FT10-T8": ("F10", "T4"),   # FT10 proxy = F10
    "FT9-FT10":("F9", "F10"),   # FT9/FT10 proxy = F9/F10
    "FZ-CZ":   ("FZ", "CZ"),
    "P3-O1":   ("P3", "O1"),
    "P4-O2":   ("P4", "O2"),
    "P7-O1":   ("T5", "O1"),
    "P7-T7":   ("T5", "T3"),
    "P8-O2":   ("T6", "O2"),
    "T7-FT9":  ("T3", "F9"),    # FT9 proxy = F9
    "T7-P7":   ("T3", "T5"),
    "T8-P8-0": ("T4", "T6"),
    "T8-P8-1": ("T4", "T6"),    # CHB-MIT lists T8-P8 twice; keep the duplicate
}


def _norm(ch):
    """'EEG Fp1' -> 'FP1', 'EKG EKG' -> 'EKGEKG'."""
    return ch.upper().replace("EEG", "").replace(" ", "").strip()


def _hms_to_sec(t):
    """'19.39.33' -> seconds since midnight (float)."""
    parts = re.split(r"[.:]", t.strip())
    h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]))
    return h * 3600 + m * 60 + s


def parse_siena_seizures(list_path):
    """Parse Seizures-list-PNxx.txt -> {edf_filename: [(start_sec, end_sec), ...]}.

    Times in the file are wall-clock (HH.MM.SS); we convert each seizure to seconds
    from its file's registration start. Midnight wrap is handled; end times are left
    to be clamped to the real EDF duration by the loader (some are typo'd past it).
    """
    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    out = {}
    # split into per-seizure blocks
    blocks = re.split(r"Seizure n\b", text)[1:]
    for b in blocks:
        fn = re.search(r"File name:\s*(\S+\.edf)", b, re.IGNORECASE)
        reg = re.search(r"Registration start time:\s*([0-9.:]+)", b, re.IGNORECASE)
        ss = re.search(r"Seizure start time:\s*([0-9.:]+)", b, re.IGNORECASE)
        se = re.search(r"Seizure end time:\s*([0-9.:]+)", b, re.IGNORECASE)
        if not (fn and reg and ss and se):
            continue
        fname = fn.group(1)
        reg0 = _hms_to_sec(reg.group(1))
        start = _hms_to_sec(ss.group(1)) - reg0
        end = _hms_to_sec(se.group(1)) - reg0
        if start < 0:
            start += 86400
        if end < 0:
            end += 86400
        out.setdefault(fname, []).append((start, end))
    return out


def load_and_window_siena(edf_path, seizures, channel_names, window_sec=WINDOW_SEC,
                          max_seizure_sec=MAX_SEIZURE_SEC):
    """Window one Siena EDF into the CHB-MIT bipolar montage `channel_names`.

    Returns (windows[n, C, T], labels[n], n_missing_channels). Channels whose
    electrodes are absent for this subject are zero-filled.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1e-3:
        raw.resample(TARGET_SFREQ)
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # (n_signals, n_samples), volts
    idx = {_norm(ch): i for i, ch in enumerate(raw.ch_names)}
    n_samples = data.shape[1]

    rows, n_missing = [], 0
    for name in channel_names:
        plus, minus = BIPOLAR_MAP.get(name, (None, None))
        if plus in idx and minus in idx:
            rows.append(data[idx[plus]] - data[idx[minus]])
        else:
            rows.append(np.zeros(n_samples, dtype=data.dtype))
            n_missing += 1
    sig = np.asarray(rows)  # (C, n_samples)

    # clamp seizure intervals to the real recording length, then cap typo'd over-long ones
    dur = n_samples / sfreq
    clamped = []
    for s, e in seizures:
        if e <= 0 or s >= dur:
            continue
        s, e = max(0.0, s), min(dur, e)
        if e - s > max_seizure_sec:
            print(f"  ~ capping over-long seizure {s:.0f}-{e:.0f}s -> {s:.0f}-{s + max_seizure_sec:.0f}s (annotation typo)")
            e = s + max_seizure_sec
        clamped.append((s, e))

    window_size = int(window_sec * sfreq)
    n_windows = n_samples // window_size
    windows, labels = [], []
    for i in range(n_windows):
        a, b = i * window_size, i * window_size + window_size
        start_sec, end_sec = a / sfreq, b / sfreq
        label = 0
        for s_start, s_end in clamped:
            if start_sec < s_end and end_sec > s_start:
                label = 1
                break
        windows.append(sig[:, a:b])
        labels.append(label)

    return np.array(windows), np.array(labels), n_missing


def available_siena_subjects(data_root=DATA_ROOT):
    """PNxx dirs that have a seizure list AND at least one EDF downloaded."""
    subs = []
    for d in sorted(glob.glob(os.path.join(data_root, "PN*"))):
        subj = os.path.basename(d)
        if glob.glob(os.path.join(d, f"Seizures-list-{subj}.txt")) and glob.glob(os.path.join(d, "*.edf")):
            subs.append(subj)
    return subs


def _resolve_edf(fname, subj_dir):
    """Map a seizure-list filename to a real EDF on disk, tolerating Siena's
    annotation typos: letter 'O' vs zero '0' (PN06's list says PNO6-1.edf), and a
    missing '-1' on single-file subjects (list says PN01.edf, file is PN01-1.edf)."""
    exact = os.path.join(subj_dir, fname)
    if os.path.exists(exact):
        return exact
    key = fname.upper().replace("O", "0")
    disk = glob.glob(os.path.join(subj_dir, "*.edf"))
    for p in disk:                                   # O<->0 normalized match
        if os.path.basename(p).upper().replace("O", "0") == key:
            return p
    stem = key[:-4]                                  # e.g. 'PN01' -> matches 'PN01-1.edf'
    for p in disk:
        if os.path.basename(p).upper().replace("O", "0").startswith(stem + "-"):
            return p
    return None


def build_siena_dataset(channel_names, subjects=None, window_sec=WINDOW_SEC,
                        data_root=DATA_ROOT, verbose=True):
    """Build (X, y, subject_ids) over Siena's seizure EDFs in the CHB-MIT montage."""
    subjects = subjects or available_siena_subjects(data_root)
    X_list, y_list, subj_list = [], [], []
    for subj in subjects:
        list_path = os.path.join(data_root, subj, f"Seizures-list-{subj}.txt")
        seizures = parse_siena_seizures(list_path)
        for fname, intervals in seizures.items():
            path = _resolve_edf(fname, os.path.join(data_root, subj))
            if path is None:
                continue
            try:
                w, lab, miss = load_and_window_siena(path, intervals, channel_names, window_sec)
            except Exception as e:
                print(f"  !! skipping {subj}/{fname}: {type(e).__name__}: {e}")
                continue
            if verbose:
                print(f"{subj}/{fname}: {w.shape[0]} windows, {int(lab.sum())} seizure"
                      + (f", {miss} ch zero-filled" if miss else ""))
            X_list.append(w.astype(np.float32))
            y_list.append(lab)
            subj_list.append(np.full(len(lab), subj))
    if not X_list:
        raise RuntimeError("No Siena data built — is data/siena populated?")
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(subj_list, axis=0)
    return X, y, subject_ids


if __name__ == "__main__":
    # quick smoke test on whatever is downloaded so far
    from src.cross_subject import common_channels
    from src.subjects import available_subjects
    chans = common_channels(available_subjects(require_seizure=True))
    print(f"CHB-MIT montage ({len(chans)}): {chans}\n")
    subs = available_siena_subjects()
    print(f"Siena subjects available: {subs}\n")
    X, y, sid = build_siena_dataset(chans, subjects=subs[:1] if subs else None)
    print(f"\nSiena (partial): {X.shape[0]} windows, {int(y.sum())} seizure, shape {X.shape}")
