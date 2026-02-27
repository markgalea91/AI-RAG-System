# src/rag_platform/api/app.py
from __future__ import annotations

from fastapi import FastAPI
from rag_platform.config.settings import get_settings
from rag_platform.common.health import (
    aggregate,
    check_milvus_uri,
    check_nim_embedding,
    check_ollama,
)

from rag_platform.api.retrieval import router as retrieval_router
app = FastAPI(title="rag_platform")
app.include_router(retrieval_router)

@app.get("/health")
def health():
    # “process is alive”
    return {"ok": True}

@app.get("/ready")
def ready():
    # “dependencies are reachable”
    s = get_settings()

    checks = [
        check_milvus_uri(s.milvus_uri),
        check_nim_embedding(s.embedding_endpoint),
        check_ollama(getattr(s, "ollama_base_url", "http://localhost:11434")),
    ]
    return aggregate(checks)