"""Swappable LLM providers for EEG report generation.

Public entry point is `src.clinical_report.build_report`. This package isolates
the LLM backend (local Foundry Local model, cloud Anthropic, or a deterministic
template) behind one interface so the backend is a config choice, not a rewrite.
"""