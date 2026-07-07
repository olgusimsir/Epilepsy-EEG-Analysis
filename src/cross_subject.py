"""
Multi-subject data pipeline for cross-subject seizure detection.

Kept separate from windowing.py / build_dataset.py (which power the single-subject
app + train.py) so those keep working unchanged. The differences here:

  * channels are selected BY NAME to a common montage shared across all subjects
    (CHB-MIT patients have different electrode montages), and reordered consistently
  * ALL seizures in a file are labeled (build_dataset.py only kept the first)
  * each window carries its subject id, so we can split by patient
"""
import os
import glob

import mne
import numpy as np

from src.parse_summary import parse_summary

WINDOW_SEC = 5
DATA_ROOT = "data/raw"


def _channels(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    return raw.ch_names


def common_channels(subjects):
    """Channel names present (by exact name) in every subject's recordings.

    Uses the first downloaded EDF of each subject as that subject's montage.
    Returned sorted so the channel order is deterministic across runs.
    """
    sets = []
    for subj in subjects:
        edfs = sorted(glob.glob(f"{DATA_ROOT}/{subj}/{subj}_*.edf"))
        if not edfs:
            raise FileNotFoundError(f"No EDF files found for {subj}")
        sets.append(set(_channels(edfs[0])))
    return sorted(set.intersection(*sets))


def load_and_window_multi(edf_path, seizures, channel_names, window_sec=WINDOW_SEC):
    """Window one recording, selecting `channel_names` (in that order) and labeling
    a window seizure if it overlaps ANY seizure interval in `seizures`."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.pick(channel_names)
    raw.reorder_channels(channel_names)  # guarantee a consistent channel order
    sfreq = raw.info["sfreq"]
    data = raw.get_data()

    window_size = int(window_sec * sfreq)
    n_windows = data.shape[1] // window_size

    windows, labels = [], []
    for i in range(n_windows):
        start_sample = i * window_size
        end_sample = start_sample + window_size
        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq

        label = 0
        for s_start, s_end in seizures:
            if start_sec < s_end and end_sec > s_start:  # overlap with any seizure
                label = 1
                break

        windows.append(data[:, start_sample:end_sample])
        labels.append(label)

    return np.array(windows), np.array(labels)


def zscore_per_channel(X):
    """Per-window, per-channel z-score: each channel of each window is scaled to
    zero mean / unit variance. Removes patient-specific amplitude and DC offset,
    which is the main obstacle to raw-waveform models generalizing across patients."""
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mean) / std


def build_multi_dataset(subjects, channel_names, window_sec=WINDOW_SEC):
    """Build (X, y, subject_ids) over every downloaded EDF of every subject."""
    X_list, y_list, subj_list = [], [], []

    for subj in subjects:
        info = parse_summary(f"{DATA_ROOT}/{subj}/{subj}-summary.txt")
        for fname, seizures in info.items():
            path = f"{DATA_ROOT}/{subj}/{fname}"
            if not os.path.exists(path):
                continue  # only use files we actually downloaded
            try:
                windows, labels = load_and_window_multi(path, seizures, channel_names, window_sec)
            except Exception as e:
                # Skip unreadable files (e.g. a failed download saved as an HTML
                # error page) instead of crashing the whole run.
                print(f"  !! skipping {subj}/{fname}: {type(e).__name__}: {e}")
                continue
            n_seiz = int(labels.sum())
            print(f"{subj}/{fname}: {windows.shape[0]} windows, {n_seiz} seizure")

            X_list.append(windows)
            y_list.append(labels)
            subj_list.append(np.full(len(labels), subj))

    # float32 (MNE returns float64) — halves memory for the stacked window tensor.
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(subj_list, axis=0)
    return X, y, subject_ids


if __name__ == "__main__":
    subjects = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08"]
    chans = common_channels(subjects)
    print(f"Common channels ({len(chans)}): {chans}\n")
    X, y, sid = build_multi_dataset(subjects, chans)
    print(f"\nTotal: {X.shape[0]} windows, {int(y.sum())} seizure")
    for subj in subjects:
        mask = sid == subj
        print(f"  {subj}: {mask.sum()} windows, {int(y[mask].sum())} seizure")