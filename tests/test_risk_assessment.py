"""Unit tests for the recording-level risk assessment, including the episode
post-processing (smoothing + merge + minimum-duration filtering).

Runs with pytest, or standalone:  python tests/test_risk_assessment.py
Only depends on numpy — no torch / mne needed, so CI stays fast.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.risk_assessment import (
    assess_recording,
    smooth,
    windows_to_episodes,
    merge_and_filter_episodes,
)


def test_smooth_damps_isolated_spike():
    probs = np.zeros(11)
    probs[5] = 1.0                      # a single-window spike
    out = smooth(probs, k=3)
    assert out[5] < 0.5                 # the spike is pulled down by its zero neighbours
    assert out.shape == probs.shape


def test_smooth_preserves_edges_without_dilution():
    probs = np.ones(6)
    out = smooth(probs, k=3)
    # a constant signal must stay constant (edges averaged over real neighbours only)
    assert np.allclose(out, 1.0)


def test_windows_to_episodes_basic():
    preds = [0, 1, 1, 0, 1, 0]         # two runs
    eps = windows_to_episodes(preds, window_sec=5)
    assert eps == [(5, 15), (20, 25)]


def test_merge_bridges_short_gap():
    # two episodes separated by a 5s gap → merged when merge_gap_sec >= 5
    eps = [(0, 20), (25, 60)]
    merged = merge_and_filter_episodes(eps, merge_gap_sec=10, min_duration_sec=10)
    assert merged == [(0, 60)]


def test_filter_drops_short_episode():
    eps = [(0, 5)]                      # a single 5s window
    out = merge_and_filter_episodes(eps, merge_gap_sec=10, min_duration_sec=10)
    assert out == []


def test_single_window_spike_is_not_high_risk():
    # one isolated high-confidence window should NOT become a High-risk episode
    probs = np.zeros(720)
    probs[300] = 0.99
    a = assess_recording(probs)
    assert a["n_episodes"] == 0
    assert a["risk_level"] != "High"


def test_sustained_seizure_is_one_high_episode():
    # a clear ~40s sustained seizure (8 consecutive windows) → one High episode
    probs = np.zeros(720)
    probs[200:208] = 0.97
    a = assess_recording(probs)
    assert a["n_episodes"] == 1
    assert a["risk_level"] == "High"
    assert a["risk_score"] > 60


def test_fragmented_seizure_merges_to_one_episode():
    # a real seizure with a brief sub-threshold dip should read as ONE episode
    probs = np.zeros(720)
    probs[200:206] = 0.95
    probs[206] = 0.10                   # brief dip
    probs[207:213] = 0.95
    a = assess_recording(probs)
    assert a["n_episodes"] == 1


def test_risk_score_is_bounded():
    probs = np.ones(720)               # pathological all-seizure input
    a = assess_recording(probs)
    assert 0 <= a["risk_score"] <= 100


def test_empty_recording_is_low():
    a = assess_recording(np.zeros(720))
    assert a["risk_level"] == "Low"
    assert a["n_episodes"] == 0
    assert a["risk_score"] == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")