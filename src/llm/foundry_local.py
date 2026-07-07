"""Local, offline LLM provider backed by Azure AI Foundry Local.

Runs entirely on-device via Foundry Local's OpenAI-compatible REST service — no
patient data leaves the machine. The model is selected by alias (e.g. "phi-3.5-mini",
"phi-4", "mistral-..."), so swapping the underlying model is a one-line config change.

API verified against Microsoft Learn "Integrate with inference SDKs" (Python),
updated 2026-06. Requires:  pip install foundry-local-sdk openai
and the Foundry Local service installed (macOS: `brew install foundrylocal`).

The manager + model load (download execution providers, download/load the model,
start the web service) is expensive, so it is done lazily once and reused.
"""
import sys

from src.llm.base import LLMProvider

DEFAULT_MODEL_ALIAS = "phi-3.5-mini"


def _progress(prefix):
    def cb(*args):
        # SDK passes either (percent) or (ep_name, percent); show whatever we get.
        msg = " ".join(str(a) for a in args)
        print(f"\r{prefix}: {msg}", end="", file=sys.stderr, flush=True)
    return cb


class FoundryLocalProvider(LLMProvider):
    name = "foundry_local"

    def __init__(self, model_alias=DEFAULT_MODEL_ALIAS, app_name="eeg_epilepsy",
                 temperature=0.2, max_tokens=700):
        self.model_alias = model_alias
        self.app_name = app_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._model_id = None

    def _ensure_ready(self):
        """Boot the local service and load the model once; reuse afterwards."""
        if self._client is not None:
            return
        import openai
        from foundry_local_sdk import Configuration, FoundryLocalManager

        config = Configuration(app_name=self.app_name)
        # The manager is a process-wide singleton; initialize it only once. A second
        # initialize() (e.g. another provider in a long-running server) raises
        # "already been initialized", so reuse the existing instance in that case.
        try:
            FoundryLocalManager.initialize(config)
        except Exception:
            pass  # already initialized in this process — reuse the singleton below
        manager = FoundryLocalManager.instance

        # First run downloads execution providers + the model (can take minutes).
        manager.download_and_register_eps(progress_callback=_progress("EP"))
        model = manager.catalog.get_model(self.model_alias)
        model.download(_progress("model"))
        model.load()
        manager.start_web_service()

        base_url = f"{manager.urls[0]}/v1"
        self._client = openai.OpenAI(base_url=base_url, api_key="none")  # key unused locally
        self._model_id = model.id

    def generate(self, system, user):
        self._ensure_ready()
        resp = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()