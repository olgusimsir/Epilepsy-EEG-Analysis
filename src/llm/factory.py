"""Provider selection.

Default is the LOCAL, offline Foundry Local provider (no data leaves the machine).
Override with the REPORT_LLM_PROVIDER env var:
    foundry_local  -> local Foundry Local model (default)
    anthropic      -> cloud Anthropic (opt-in; sends data off-device)
    template       -> deterministic offline report (returns None here)

REPORT_LLM_MODEL overrides the model alias/id (e.g. phi-4, mistral-7b-...).
"""
import os

from src.llm.base import LLMProvider

DEFAULT_PROVIDER = "foundry_local"

# Foundry Local boots a process-wide singleton manager and does an expensive
# model load, so its provider must be created ONCE per process and reused. Caching
# here keeps the warmed client alive across reports (and avoids re-initializing the
# singleton, which raises "already been initialized" on the second call).
_FOUNDRY_PROVIDERS = {}


def get_provider(name=None, api_key=None, model=None):
    """Return an LLMProvider, or None to signal the deterministic template path."""
    name = (name or os.environ.get("REPORT_LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
    model = model or os.environ.get("REPORT_LLM_MODEL")

    if name == "template":
        return None
    if name == "foundry_local":
        from src.llm.foundry_local import FoundryLocalProvider, DEFAULT_MODEL_ALIAS
        alias = model or DEFAULT_MODEL_ALIAS
        if alias not in _FOUNDRY_PROVIDERS:
            _FOUNDRY_PROVIDERS[alias] = FoundryLocalProvider(model_alias=alias)
        return _FOUNDRY_PROVIDERS[alias]
    if name == "anthropic":
        from src.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model or "claude-sonnet-4-6", api_key=api_key)
    raise ValueError(f"Unknown REPORT_LLM_PROVIDER: {name!r}")