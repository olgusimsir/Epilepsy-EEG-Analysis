"""Provider interface — every LLM backend implements `generate(system, user)`."""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Return the model's text completion for the given system + user prompt."""
        raise NotImplementedError