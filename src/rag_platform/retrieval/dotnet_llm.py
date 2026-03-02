# src/rag_platform/retrieval/dotnet_llm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass
class DotNetLlmClientConfig:
    base_url: str                       # e.g. "http://localhost:5000"
    timeout_s: float = 180.0
    verify_tls: bool = True
    extra_headers: Optional[Dict[str, str]] = None

    # The fixed provider/model values you want to send to your API
    provider: str = "Ollama"
    model: str = "nemotron-3-nano:30b"
    is_chat: bool = False


class DotNetLlmClient:
    """
    Calls your .NET LLM Integration:
      - POST /Authenticate   (handled by DotNetAuthenticator)
      - POST /SendGenerateRequest (here)
    """

    def __init__(self, cfg: DotNetLlmClientConfig, authenticator):
        self.cfg = cfg
        self.authenticator = authenticator

    def _build_headers(self, token: str) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        eh = self.cfg.extra_headers
        if eh is None:
            return headers

        # ✅ Strict type check to prevent your current crash
        if not isinstance(eh, dict):
            raise TypeError(
                f"DotNetLlmClientConfig.extra_headers must be a dict[str,str], got {type(eh).__name__}: {eh!r}"
            )

        # Optional: force string keys/values
        for k, v in eh.items():
            headers[str(k)] = str(v)

        return headers

    def send_generate_request(self, *, prompt_text: str, request_text: str) -> Dict[str, Any]:
        """
        POST /SendGenerateRequest
        Body:
          {
            "apiKey": "...",
            "provider": "...",
            "model": "...",
            "isChat": true,
            "promptText": "...",
            "requestText": "..."
          }
        Returns ResponseModel JSON as dict.
        """
        token = self.authenticator.get_token()

        url = self.cfg.base_url.rstrip("/") + "/LlmService/SendGenerateRequest"
        headers = self._build_headers(token)

        payload = {
            "ApiKey": self.authenticator.cfg.guid,                  # <-- match your BaseRequestModel.ApiKey
            "Provider": self.cfg.provider,
            "Model": self.cfg.model,
            "IsChat": self.cfg.is_chat,
            "PromptText": prompt_text,
            "RequestText": request_text,
        }

        with httpx.Client(timeout=self.cfg.timeout_s, verify=self.cfg.verify_tls) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if not isinstance(data, dict):
            raise RuntimeError(f"SendGenerateRequest expected JSON object, got: {type(data).__name__}")

        return data