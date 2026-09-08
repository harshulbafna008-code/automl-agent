"""LLM client backed by a local Ollama server (https://ollama.com).

No API key required. Ollama must be running (`ollama serve`, or it's
already running as a background service after install) and the requested
model must be pulled (`ollama pull <model>`).
"""

from __future__ import annotations

import logging

import requests

from .base import LLMClient

logger = logging.getLogger("automl_agent.llm.ollama")


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            tags = [m.get("name", "") for m in resp.json().get("models", [])]
            # Accept exact match or "name" without a ":tag" suffix matching.
            available = any(self.model == t or t.startswith(self.model.split(":")[0]) for t in tags)
            if not available:
                logger.warning(
                    "Ollama is running but model '%s' was not found in `ollama list`. "
                    "Run `ollama pull %s` first.",
                    self.model,
                    self.model,
                )
            return True  # server reachable; model availability is a soft warning
        except requests.RequestException as exc:
            logger.warning("Could not reach Ollama at %s: %s", self.host, exc)
            return False

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
