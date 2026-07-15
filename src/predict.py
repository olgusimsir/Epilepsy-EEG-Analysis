"""Unified inference for either trained model.

Handles both checkpoint shapes:
  * single-patient: SeizureCNN, all native channels (no `channels` key)
  * cross-subject:  SeizureCNN2, a fixed channel montage selected by name
The checkpoint carries everything needed (arch, channels, norm stats), so the
caller just picks a path.
"""
import json
import os

import numpy as np
import torch

from src.model import SeizureCNN, SeizureCNN2

WINDOW_SEC = 5


def normalize_windows(windows, ckpt):
    """Preprocess windows to match how the model was trained.

    Two modes, chosen by the checkpoint's `norm_mode` (default 'global' for backward
    compatibility with existing models):
      * 'global'      — subtract one dataset-wide mean/std (stored in the checkpoint).
      * 'per_window'  — z-score EACH window's EACH channel to zero-mean/unit-var. This
        removes per-recording amplitude/DC offset, which is what lets a raw-waveform
        model transfer across datasets/hardware (used for cross-dataset robustness).
    """
    if ckpt.get("norm_mode") == "per_window":
        m = windows.mean(axis=2, keepdims=True)
        s = windows.std(axis=2, keepdims=True) + 1e-8
        return (windows - m) / s
    return (windows - ckpt["norm_mean"]) / ckpt["norm_std"]


def load_checkpoint(path):
    """Return (model, ckpt). Builds the right architecture from ckpt['arch'].

    If a sidecar `calibration.json` sits next to the checkpoint, its temperature is
    attached as ckpt['temperature'] so inference reports calibrated probabilities.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "SeizureCNN")
    n_ch = ckpt.get("n_channels", 23)
    model = SeizureCNN2(n_ch) if arch == "SeizureCNN2" else SeizureCNN(n_ch)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ckpt.setdefault("temperature", 1.0)
    calib = os.path.join(os.path.dirname(path), "calibration.json")
    if os.path.exists(calib):
        try:
            with open(calib) as f:
                ckpt["temperature"] = float(json.load(f).get("temperature", 1.0))
        except Exception:
            pass
    return model, ckpt


TARGET_SFREQ = 256


def _montage_data(edf_path, channels):
    """Load an EDF as (len(channels), samples) in the model's montage at 256 Hz.

    Montage-agnostic — accepts either input format:
      * CHB-MIT style: the bipolar channels (FP1-F7, ...) already exist by name → pick them.
      * Referential style (e.g. Siena: EEG Fp1, ...): DERIVE each bipolar channel by
        subtraction via siena.BIPOLAR_MAP (T3->T7 etc., F9/F10 proxy for FT9/FT10; any
        electrode this recording lacks is zero-filled).
    Resamples to 256 Hz so the window length matches training. Raises ValueError if the
    montage is unrecognizable (nothing could be built)."""
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1e-3:
        raw.resample(TARGET_SFREQ)
    have = set(raw.ch_names)
    if all(c in have for c in channels):                 # CHB-MIT format — direct pick
        raw.pick(channels); raw.reorder_channels(channels)
        return raw.get_data(), raw.info["sfreq"]
    from src.siena import BIPOLAR_MAP, _norm                # referential — derive bipolar
    data = raw.get_data()
    idx = {_norm(c): i for i, c in enumerate(raw.ch_names)}
    rows, built = [], 0
    for name in channels:
        plus, minus = BIPOLAR_MAP.get(name, (None, None))
        if plus in idx and minus in idx:
            rows.append(data[idx[plus]] - data[idx[minus]]); built += 1
        else:
            rows.append(np.zeros(data.shape[1], dtype=float))
    if built == 0:
        raise ValueError("unrecognized EEG montage — expected CHB-MIT bipolar channels "
                         "or standard 10-20 referential electrodes")
    return np.asarray(rows), raw.info["sfreq"]


def _window(edf_path, ckpt, window_sec):
    """Window the recording into the model's montage (montage-agnostic)."""
    channels = ckpt.get("channels")
    if not channels:  # single-patient model — all native channels
        from src.windowing import load_and_window
        windows, _ = load_and_window(edf_path, window_sec=window_sec)
        return windows
    data, sfreq = _montage_data(edf_path, channels)
    ws = int(window_sec * sfreq)
    n = data.shape[1] // ws
    if n == 0:
        return np.empty((0, len(channels), ws))
    return np.stack([data[:, i * ws:(i + 1) * ws] for i in range(n)])


def _channel_data(edf_path, ckpt):
    """Load the recording as (channels, samples) in the model's montage at 256 Hz."""
    channels = ckpt.get("channels")
    if not channels:
        import mne
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        return raw.get_data(), raw.info["sfreq"]
    return _montage_data(edf_path, channels)


def _score(model, X, temperature=1.0):
    """Batched inference over a (n, channels, samples) tensor.

    Applies temperature scaling to the logits before the sigmoid, so the returned
    probabilities are calibrated (temperature=1.0 is the raw, uncalibrated output).
    """
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):  # batch so long recordings don't blow up memory
            logits = model(X[i:i + 256]).squeeze(1)
            probs.append(torch.sigmoid(logits / temperature))
    return torch.cat(probs).numpy() if probs else np.array([])


def _overlap_probs(model, ckpt, edf_path, window_sec, overlap):
    """Score OVERLAPPING windows (stride = window_sec*(1-overlap)), then reduce the
    dense scores back onto the standard non-overlapping window grid by AVERAGING, for
    each base window, the probabilities of the overlapping windows that cover it.

    This catches seizures that straddle a window boundary (a real gain over rigid
    non-overlapping windows) without changing the downstream grid, episode-timing
    math, risk scoring, or the frontend — probs come back the same length as before.

    Averaging (rather than max) was chosen empirically: max over-triggered and added
    false-positive episodes on clean recordings, while the mean keeps precision and
    still benefits from the denser sampling.
    """
    data, sfreq = _channel_data(edf_path, ckpt)
    window_size = int(window_sec * sfreq)
    n_base = data.shape[1] // window_size
    if window_size <= 0 or n_base == 0:
        return np.array([])

    stride = max(1, int(round(window_size * (1.0 - overlap))))
    starts = list(range(0, data.shape[1] - window_size + 1, stride))
    wins = np.stack([data[:, s:s + window_size] for s in starts])

    X = normalize_windows(wins, ckpt)
    X = torch.tensor(X, dtype=torch.float32)
    dense = _score(model, X, ckpt.get("temperature", 1.0))

    summed = np.zeros(n_base)
    counts = np.zeros(n_base)
    for p, s in zip(dense, starts):
        j0 = s // window_size
        j1 = (s + window_size - 1) // window_size
        for j in range(j0, min(j1, n_base - 1) + 1):
            summed[j] += p
            counts[j] += 1
    return np.where(counts > 0, summed / np.maximum(counts, 1), 0.0)


def predict_recording(model, ckpt, edf_path, threshold=0.5, window_sec=WINDOW_SEC,
                      overlap=0.0):
    """Return (probs, preds) — per-window seizure probability and thresholded labels.

    overlap: 0.0 = classic non-overlapping windows (default, unchanged). A value in
    (0,1) scores overlapping windows and reduces them to the same base grid — better
    boundary sensitivity, identical output shape.
    """
    if overlap and overlap > 0:
        probs = _overlap_probs(model, ckpt, edf_path, window_sec, overlap)
    else:
        windows = _window(edf_path, ckpt, window_sec)
        X = normalize_windows(windows, ckpt)
        X = torch.tensor(X, dtype=torch.float32)
        probs = _score(model, X, ckpt.get("temperature", 1.0))
    return probs, (probs > threshold).astype(int)