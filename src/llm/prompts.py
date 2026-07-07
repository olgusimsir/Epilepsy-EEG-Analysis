"""EEG report prompt — shared by every LLM provider so the clinical wording lives
in exactly one place. Written to be explicit and well-delimited, which matters for
smaller local models (Phi-3.5/Phi-4) that follow loose instructions less reliably
than large cloud models.
"""
import json

SYSTEM_PROMPT = (
    "You are a clinical neurophysiology assistant that drafts descriptive EEG "
    "analysis reports in the style of an expert neurologist's impression. You are "
    "given the JSON output of an automated seizure-detection model for ONE EEG "
    "recording. Write a concise, professional English report with exactly these "
    "three sections, each as a short paragraph:\n"
    "  Findings — what the automated analysis observed (epochs, episodes, timing, "
    "confidence).\n"
    "  Risk Assessment — interpret the risk level and score.\n"
    "  Recommendation — next clinical step.\n\n"
    "Rules:\n"
    "- Base every statement ONLY on the provided data. Do not invent values.\n"
    "- This is an automated DECISION-SUPPORT draft, not a diagnosis. State this and "
    "recommend review by a qualified neurologist.\n"
    "- Do not output JSON, code, or markdown headers — plain labeled paragraphs only."
)


def build_user_message(assessment, filename, context_chunks=None):
    """Turn the structured assessment dict into the user-turn prompt.

    If `context_chunks` (RAG retrieval results) are given, they are included as
    reference material the model should ground its wording and standards in.
    """
    parts = [
        f"Automated EEG analysis for recording '{filename}':\n\n"
        + json.dumps(assessment, indent=2)
    ]
    if context_chunks:
        refs = "\n\n".join(f"[{c['source']}] {c['text']}" for c in context_chunks)
        parts.append(
            "Reference material — use it for terminology, report structure, and "
            "clinical standards; cite the bracketed source name when you rely on it:\n\n"
            + refs
        )
    parts.append("Write the descriptive EEG analysis report now.")
    return "\n\n---\n\n".join(parts)