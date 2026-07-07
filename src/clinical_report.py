"""Public entry point for EEG report generation.

Thin orchestrator: build the EEG prompt, pick a provider (local Foundry Local by
default, see src/llm/factory.py), generate, and fall back to a deterministic
template if the provider is unavailable. Callers (app.py, CLI) only need this.
"""
from src.llm.prompts import SYSTEM_PROMPT, build_user_message
from src.llm.template import template_report
from src.llm.factory import get_provider


def build_report(assessment, filename, provider=None, api_key=None, model=None,
                 use_rag=False, knowledge_path="data/knowledge", k=4):
    """Return (report_text, source). `source` names what produced the report.

    provider: override the LLM backend ("foundry_local" | "anthropic" | "template").
              Defaults to REPORT_LLM_PROVIDER env (local Foundry Local).
    api_key:  only used by the opt-in Anthropic provider.
    model:    override the model id/alias (e.g. "phi-3.5-mini", "phi-4").
    use_rag:  if True, retrieve reference passages from `knowledge_path` and ground
              the report in them (fully local).
    """
    context = None
    if use_rag:
        from src.rag.index import build_retriever, query_from_assessment
        retriever = build_retriever(knowledge_path)
        context = retriever.retrieve(query_from_assessment(assessment), k=k)

    prov = get_provider(provider, api_key=api_key, model=model)
    tag = "+rag" if context else ""

    if prov is None:  # template selected (template does not consume context)
        return template_report(assessment, filename), "template"

    try:
        user = build_user_message(assessment, filename, context_chunks=context)
        return prov.generate(SYSTEM_PROMPT, user), prov.name + tag
    except Exception as e:
        # Any provider failure (service down, missing package, network) -> offline template.
        return (
            template_report(assessment, filename) + f"\n\n[{prov.name} unavailable: {e}]",
            f"template ({prov.name} error)",
        )


if __name__ == "__main__":
    import numpy as np
    from src.risk_assessment import assess_recording

    demo = np.zeros(720)
    demo[600:609] = [0.6, 0.9, 0.98, 0.99, 0.95, 0.88, 0.7, 0.55, 0.4]
    a = assess_recording(demo)
    text, source = build_report(a, "demo.edf")
    print(f"[source: {source}]\n")
    print(text)