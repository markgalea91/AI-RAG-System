from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


GUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


@dataclass
class DotNetApiConfig:
    base_url: str                 # e.g. "http://localhost:5000"
    timeout_s: float = 30.0
    verify_tls: bool = True
    extra_headers: Optional[Dict[str, str]] = None


class DotNetApiClient:
    def __init__(self, cfg: DotNetApiConfig):
        self.cfg = cfg

    def _client(self) -> httpx.Client:
        headers = {}
        if self.cfg.extra_headers:
            headers.update(self.cfg.extra_headers)
        return httpx.Client(timeout=self.cfg.timeout_s, verify=self.cfg.verify_tls, headers=headers)

    def add_client(self, client_name: str) -> Dict[str, Any]:
        """
        POST /AddClient?request=<client_name>
        Returns ResponseModel as dict.
        """
        url = f"{self.cfg.base_url.rstrip('/')}/AddClient"
        with self._client() as c:
            r = c.post(url, params={"request": client_name})
            r.raise_for_status()
            return r.json()

    def extract_client_key(self, addclient_response: Dict[str, Any]) -> str:
        """
        ResponseModel contains: AssistantResponse = "Client created. Client Key: <GUID>"
        """
        msg = (addclient_response.get("AssistantResponse") or "").strip()
        m = GUID_RE.search(msg)
        if not m:
            raise RuntimeError(f"Could not extract client key GUID from AssistantResponse: {msg!r}")
        return m.group(0)

    def authenticate(self, client_key_guid: str) -> str:
        """
        POST /Authenticate?request=<client_key_guid>
        Returns JWT string if valid, else error string.
        """
        url = f"{self.cfg.base_url.rstrip('/')}/Authenticate"
        with self._client() as c:
            r = c.post(url, params={"request": client_key_guid})
            r.raise_for_status()
            token = r.text.strip()

        # crude validation: JWT has 3 dot-separated parts
        if token.count(".") != 2:
            raise RuntimeError(f"Authenticate did not return a JWT. Response was: {token!r}")
        return token