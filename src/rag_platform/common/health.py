# src/rag_platform/common/health.py
from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx


@dataclass
class CheckResult:
    ok: bool
    name: str
    detail: str = ""
    latency_ms: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


def _latency_ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def check_tcp(host: str, port: int, *, timeout_s: float = 1.5, name: str = "tcp") -> CheckResult:
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return CheckResult(ok=True, name=name, detail=f"{host}:{port}", latency_ms=_latency_ms(t0))
    except Exception as e:
        return CheckResult(ok=False, name=name, detail=f"{host}:{port} - {type(e).__name__}: {e}", latency_ms=_latency_ms(t0))


def _parse_host_port_from_url(url: str, default_port: int) -> tuple[str, int]:
    u = urlparse(url)
    host = u.hostname or "localhost"
    port = u.port or default_port
    return host, port


def check_http_get(url: str, *, timeout_s: float = 2.5, name: str = "http") -> CheckResult:
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(url)
        ok = 200 <= r.status_code < 300
        detail = f"{r.status_code} {url}"
        return CheckResult(ok=ok, name=name, detail=detail, latency_ms=_latency_ms(t0))
    except Exception as e:
        return CheckResult(ok=False, name=name, detail=f"{url} - {type(e).__name__}: {e}", latency_ms=_latency_ms(t0))


def check_ollama(base_url: str = "http://localhost:11434", *, timeout_s: float = 2.5) -> CheckResult:
    # Ollama exposes /api/tags
    return check_http_get(f"{base_url.rstrip('/')}/api/tags", timeout_s=timeout_s, name="ollama")


def check_nim_embedding(endpoint: str, *, timeout_s: float = 2.5) -> CheckResult:
    """
    For OpenAI-compatible endpoints, /v1/models often works.
    If your NIM doesn't support it, fall back to TCP check.
    """
    endpoint = endpoint.rstrip("/")
    # Try /v1/models first
    models_url = f"{endpoint}/models" if endpoint.endswith("/v1") else f"{endpoint}/v1/models"
    res = check_http_get(models_url, timeout_s=timeout_s, name="nim_embedding_http")

    if res.ok:
        return res

    # Fallback: just check port is open
    try:
        host, port = _parse_host_port_from_url(endpoint, default_port=80)
        tcp = check_tcp(host, port, timeout_s=timeout_s, name="nim_embedding_tcp")
        # keep the original HTTP failure detail as extra context
        tcp.extra = {"http_attempt": res.detail}
        return tcp
    except Exception:
        return res


def check_milvus_uri(milvus_uri: str, *, timeout_s: float = 2.0) -> CheckResult:
    """
    Handles URIs like:
      tcp://localhost:19530
      http://localhost:19530
    We do a TCP connect as a fast readiness signal.
    (You can add a deeper pymilvus ping later if you want.)
    """
    host, port = _parse_host_port_from_url(milvus_uri, default_port=19530)
    return check_tcp(host, port, timeout_s=timeout_s, name="milvus_tcp")


def aggregate(results: list[CheckResult]) -> Dict[str, Any]:
    ok = all(r.ok for r in results)
    return {
        "ok": ok,
        "checks": [
            {
                "name": r.name,
                "ok": r.ok,
                "detail": r.detail,
                "latency_ms": r.latency_ms,
                **({"extra": r.extra} if r.extra else {}),
            }
            for r in results
        ],
    }