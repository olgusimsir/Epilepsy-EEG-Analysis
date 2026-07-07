"""Cloud LLM provider (Anthropic claude-sonnet-4-6).

OPT-IN AND OFF BY DEFAULT. Selecting this provider sends the structured EEG
findings to Anthropic's cloud API — data leaves the machine. Do not use it with
real patient data; it exists for non-PHI demos and comparison only. The default
provider is the local, offline Foundry Local one.
"""
from src.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model="claude-sonnet-4-6", api_key=None, max_tokens=700):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def generate(self, system, user):
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()