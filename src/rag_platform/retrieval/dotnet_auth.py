from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import httpx


@dataclass
class DotNetAuthConfig:
    base_url: str                 # e.g. "https://api.yourdomain.com"
    guid: str                     # client key (GUID) to send as `request`
    timeout_s: float = 30.0
    verify_tls: bool = True
    extra_headers: Optional[Dict[str, str]] = None

    # Fallback TTL if we can't infer exp from JWT
    default_ttl_s: int = 60 * 60 * 24 * 30  # ~30 days


class DotNetAuthenticator:
    """
    Fetches and caches JWT from AllowAnonymous Authenticate endpoint.
    """
    def __init__(self, cfg: DotNetAuthConfig):
        self.cfg = cfg
        self._token: Optional[str] = None
        self._exp_epoch: float = 0.0  # unix epoch when token expires

    def invalidate(self) -> None:
        self._token = None
        self._exp_epoch = 0.0

    def _is_valid(self) -> bool:
        # refresh 60s before expiry
        return bool(self._token) and (time.time() < (self._exp_epoch - 60))

    def get_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh and self._is_valid():
            return self._token  # type: ignore[return-value]

        token, ttl_s = self._authenticate()
        self._token = token
        self._exp_epoch = time.time() + ttl_s
        return token

    def _authenticate(self) -> Tuple[str, int]:
        headers = {}
        if self.cfg.extra_headers:
            headers.update(self.cfg.extra_headers)

        url = self.cfg.base_url.rstrip("/") + "/Authenticate"

        # ✅ ASP.NET Core binding-friendly:
        # POST /Authenticate?request=<guid>
        params = {"request": self.cfg.guid}

        with httpx.Client(timeout=self.cfg.timeout_s, verify=self.cfg.verify_tls) as client:
            resp = client.post(url, headers=headers, params=params)
            resp.raise_for_status()

            token = resp.text.strip()
            if not token:
                raise RuntimeError(f"Authenticate returned empty body: {resp.text[:300]!r}")

            # If you want: decode JWT exp claim (optional).
            ttl_s = self.cfg.default_ttl_s
            return token, ttl_s