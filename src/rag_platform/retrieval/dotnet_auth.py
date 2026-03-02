# src/rag_platform/llm_clients/dotnet_auth.py
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from rag_platform.retrieval.dotnet_client import DotNetApiClient


@dataclass(frozen=True)
class DotNetAuthConfig:
    """
    Auth cache config.
    """
    guid: str  # client key (GUID) used to authenticate
    # If we can't infer exp from JWT, fallback TTL:
    default_ttl_s: int = 60 * 60 * 24 * 30  # ~30 days
    # refresh a bit before expiry
    refresh_skew_s: int = 60


class DotNetAuthenticator:
    """
    Token cache that uses DotNetApiClient.authenticate(guid).

    - No httpx here
    - No URLs here
    - Just caching + (optional) JWT exp parsing
    """
    def __init__(self, *, client: DotNetApiClient, cfg: DotNetAuthConfig):
        self.client = client
        self.cfg = cfg
        self._token: Optional[str] = None
        self._exp_epoch: float = 0.0

    def invalidate(self) -> None:
        self._token = None
        self._exp_epoch = 0.0

    def _is_valid(self) -> bool:
        if not self._token:
            return False
        return time.time() < (self._exp_epoch - self.cfg.refresh_skew_s)

    def get_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh and self._is_valid():
            return self._token  # type: ignore[return-value]

        token, ttl_s = self._authenticate()
        self._token = token
        self._exp_epoch = time.time() + ttl_s
        return token

    def _authenticate(self) -> Tuple[str, int]:
        token = self.client.authenticate(self.cfg.guid)
        ttl_s = self._ttl_from_jwt(token) or self.cfg.default_ttl_s
        return token, ttl_s

    def _ttl_from_jwt(self, jwt: str) -> Optional[int]:
        """
        Best-effort: decode JWT payload (without verifying signature) and use exp claim.
        Returns TTL seconds or None if not available/parsable.
        """
        try:
            parts = jwt.split(".")
            if len(parts) != 3:
                return None

            payload_b64 = parts[1]
            # Add padding for base64url
            padding = "=" * (-len(payload_b64) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
            payload = json.loads(payload_bytes.decode("utf-8", errors="ignore"))

            exp = payload.get("exp")
            if not isinstance(exp, (int, float)):
                return None

            ttl = int(exp - time.time())
            if ttl <= 0:
                return None
            return ttl
        except Exception:
            return None