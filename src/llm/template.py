"""Deterministic, no-LLM fallback report.

Always available and fully offline. Used when REPORT_LLM_PROVIDER=template, or when
the selected provider is unavailable (service down, package missing, etc.). Works
directly from the structured assessment dict rather than from a prompt.
"""


def template_report(assessment, filename):
    a = assessment
    episodes = ", ".join(
        f"{e['start_sec']}-{e['end_sec']}s" for e in a["episodes"]
    ) or "none"
    rec = (
        "Clinical correlation and expert EEG review are recommended."
        if a["n_abnormal_windows"]
        else "No epileptiform activity was flagged; routine correlation advised."
    )
    return (
        f"EEG ANALYSIS REPORT — {filename}\n"
        f"(Automated decision-support draft — not a diagnosis. "
        f"Review by a qualified neurologist is required.)\n\n"
        f"Findings:\n"
        f"  Analyzed {a['n_windows']} consecutive 5-second epochs. "
        f"{a['n_abnormal_windows']} epoch(s) ({a['pct_abnormal']}%) showed "
        f"seizure-like activity across {a['n_episodes']} distinct episode(s) "
        f"(total {a['abnormal_seconds']}s). Detected episodes: {episodes}. "
        f"Peak model confidence: {a['peak_confidence']:.2f}.\n\n"
        f"Risk Assessment:\n"
        f"  Epilepsy risk level: {a['risk_level']} (score {a['risk_score']}/100).\n\n"
        f"Recommendation:\n  {rec}"
    )