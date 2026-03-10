# src/rag_platform/llm_clients/dotnet_client.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

GUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


@dataclass(frozen=True)
class DotNetApiConfig:
    """
    .NET API base config.
    """
    base_url: str                 # e.g. "http://localhost:5000/Client"
    timeout_s: float = 30.0
    verify_tls: bool = True
    extra_headers: Optional[Dict[str, str]] = None


class DotNetApiClient:
    """
    Thin HTTP client around your .NET Core Integration API.

    Assumptions (based on your controllers/managers):
      - POST /AddClient expects [FromBody] string request (client name)
      - POST /Authenticate expects [FromBody] string request (client key GUID)
      - Authenticate returns a plain-text JWT string on success (not JSON)

    If your controller is mounted under /Client, set base_url accordingly, e.g.:
      base_url="http://localhost:5000/Client"
    """
    def __init__(self, cfg: DotNetApiConfig):
        self.cfg = cfg

    def _headers(self, *, token: Optional[str] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.cfg.extra_headers:
            headers.update(self.cfg.extra_headers)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.cfg.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _post_body_string(self, path: str, body: str, *, token: Optional[str] = None) -> httpx.Response:
        """
        Posts a JSON string body (so ASP.NET Core [FromBody] string request binds cleanly).

        Example JSON payload for body="abc" is: "abc"
        """
        url = self._url(path)
        headers = self._headers(token=token)

        with httpx.Client(timeout=self.cfg.timeout_s, verify=self.cfg.verify_tls, headers=headers) as c:
            # IMPORTANT: json=<string> -> sends a JSON string in the request body.
            resp = c.post(url, json=body)
            resp.raise_for_status()
            return resp

    # -------------------------
    # Public endpoints
    # -------------------------

    def add_client(self, client_name: str) -> Dict[str, Any]:
        """
        POST /AddClient   body: "<client_name>"
        Returns ResponseModel JSON as dict.
        """
        resp = self._post_body_string("AddClient", client_name)
        # ResponseModel is JSON
        return resp.json()

    def extract_client_key(self, addclient_response: Dict[str, Any]) -> str:
        """
        ResponseModel contains: AssistantResponse = "Client created. Client Key: <GUID>"
        Extract and return that GUID.
        """
        msg = (addclient_response.get("AssistantResponse") or "").strip()
        m = GUID_RE.search(msg)
        if not m:
            raise RuntimeError(f"Could not extract client key GUID from AssistantResponse: {msg!r}")
        return m.group(0)

    def authenticate(self, client_key_guid: str) -> str:
        """
        POST /Authenticate  body: "<client_key_guid>"
        Returns JWT string on success (plain text).
        """
        resp = self._post_body_string("Client/Authenticate", client_key_guid)
        token = resp.text.strip()

        # crude validation: JWT has 3 dot-separated parts
        if token.count(".") != 2:
            raise RuntimeError(f"Authenticate did not return a JWT. Response was: {token!r}")
        return token