from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        url = f"{self.base_url}/api/generate"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            try:
                details = e.read().decode("utf-8", errors="replace")
            except Exception:
                details = str(e)
            raise RuntimeError(f"Ollama HTTP error: {e.code} {e.reason}: {details}")
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        try:
            data = json.loads(body)
        except Exception:
            raise RuntimeError(f"Ollama invalid JSON response: {body[:4000]}")

        text = data.get("response")
        if not isinstance(text, str):
            raise RuntimeError(f"Ollama unexpected response schema: {data}")
        return text
