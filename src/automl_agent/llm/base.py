"""Abstract interface for an LLM backend.

Keeping this thin means swapping Ollama for another local runtime (LM
Studio, vLLM, llama.cpp server) or a cloud API later only requires a new
implementation of this class, with zero changes to the agent logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Minimal chat-completion interface used by the agent."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        """Return the raw text completion for a single-turn chat exchange.

        Parameters
        ----------
        system_prompt: instructions describing the assistant's role/format.
        user_prompt: the actual task/content for this call.
        json_mode: if True, the backend should try to constrain output to
            valid JSON (support varies by backend; callers must still
            validate/parse defensively).
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Best-effort health check. Override in subclasses. Default: True."""
        return True

    @property
    def name(self) -> Optional[str]:
        return getattr(self, "model", None)
