"""
Recording-level epilepsy risk assessment.

The CNN produces a seizure probability per 5-second window. This module aggregates
those per-window probabilities into a single, recording-level assessment: summary
statistics, detected episodes, a 0-100 risk score, and a categorical risk level.

This is a decision-support heuristic, NOT a diagnosis. The score is a transparent
combination of detection burden and confidence, designed to be explainable to a
clinician — not a calibrated clinical probability.
"""
import numpy as np

# Episode post-processing defaults (seconds). Raw per-window thresholding is noisy:
# a single 5 s window flips to "seizure" on a transient, and a real seizure with a
# brief sub-threshold dip fragments into several "episodes". Smoothing + merging +
# a minimum duration turn the raw window flags into clinically meaningful events.
SMOOTH_K = 3            # centered moving-average window (in windows) applied to probs
MERGE_GAP_SEC = 10      # merge episodes separated by a gap no longer than this
MIN_EPISODE_SEC = 10    # drop episodes shorter than this (single-window false spikes)


def smooth(probs, k=SMOOTH_K):
    """Centered moving average that damps isolated single-window spikes.

    Edges are averaged over available neighbours only (no zero-dilution), so a
    seizure at the very start/end of the recording is not artificially suppressed.
    """
    probs = np.asarray(probs, dtype=float)
    if k <= 1 or probs.size < k:
        return probs
    kernel = np.ones(k)
    summed = np.convolve(probs, kernel, mode="same")
    counts = np.convolve(np.ones_like(probs), kernel, mode="same")
    return summed / counts


def windows_to_episodes(preds, window_sec):
    """Collapse consecutive positive windows into (start_sec, end_sec) episodes."""
    episodes = []
    start = None
    for i, p in enumerate(preds):
        if p and start is None:
            start = i
        elif not p and start is not None:
            episodes.append((start * window_sec, i * window_sec))
            start = None
    if start is not None:
        episodes.append((start * window_sec, len(preds) * window_sec))
    return episodes


def merge_and_filter_episodes(episodes, merge_gap_sec=MERGE_GAP_SEC,
                              min_duration_sec=MIN_EPISODE_SEC):
    """Merge episodes separated by a short sub-threshold gap, then drop episodes
    shorter than `min_duration_sec`. Reduces fragmentation and single-window
    false positives."""
    if not episodes:
        return []
    merged = [list(episodes[0])]
    for s, e in episodes[1:]:
        if s - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = e            # bridge the brief dip
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_duration_sec]


def assess_recording(probs, window_sec=5, threshold=0.5, smooth_k=SMOOTH_K,
                     merge_gap_sec=MERGE_GAP_SEC, min_episode_sec=MIN_EPISODE_SEC):
    """Turn per-window seizure probabilities into a recording-level assessment dict."""
    raw = np.asarray(probs, dtype=float)
    smoothed = smooth(raw, smooth_k)
    preds = smoothed > threshold
    n_windows = int(len(raw))
    n_abnormal = int(preds.sum())
    pct_abnormal = round(100 * n_abnormal / n_windows, 1) if n_windows else 0.0
    peak = float(raw.max()) if n_windows else 0.0        # true strongest window confidence
    mean = float(raw.mean()) if n_windows else 0.0

    episodes = windows_to_episodes(preds, window_sec)
    episodes = merge_and_filter_episodes(episodes, merge_gap_sec, min_episode_sec)
    n_episodes = len(episodes)
    abnormal_seconds = sum(e - s for s, e in episodes)

    # Categorical risk level — rule-based and explainable.
    #   High:     clear seizure-like activity (an episode at high confidence)
    #   Moderate: some abnormal windows, but lower confidence or isolated
    #   Low:      no abnormal windows detected
    if n_episodes >= 1 and peak >= 0.80:
        risk_level = "High"
    elif n_abnormal > 0:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # 0-100 risk score — a transparent blend of peak confidence (most weight),
    # number of distinct episodes, and overall burden. Monotonic in each factor.
    score = (
        70 * peak                              # confidence of the strongest detection
        + 20 * min(n_episodes, 3) / 3          # multiple events raise concern
        + 10 * min(pct_abnormal, 5) / 5        # sustained burden raises concern
    )
    risk_score = int(round(min(100, score)))

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "n_windows": n_windows,
        "n_abnormal_windows": n_abnormal,
        "pct_abnormal": pct_abnormal,
        "n_episodes": n_episodes,
        "abnormal_seconds": int(abnormal_seconds),
        "peak_confidence": round(peak, 3),
        "mean_confidence": round(mean, 3),
        "episodes": [
            {"start_sec": int(s), "end_sec": int(e)} for s, e in episodes
        ],
    }


if __name__ == "__main__":
    # Quick self-check on synthetic probabilities.
    demo = np.zeros(720)
    demo[600:609] = [0.6, 0.9, 0.98, 0.99, 0.95, 0.88, 0.7, 0.55, 0.4]  # one episode
    import json
    print(json.dumps(assess_recording(demo), indent=2))