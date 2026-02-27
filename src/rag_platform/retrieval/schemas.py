# src/rag_platform/retrieval/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=50)


class RetrievedChunk(BaseModel):
    text: str
    source_name: str = "unknown_source"
    page_number: Optional[int] = None
    score: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None  # optional debug payload


class RetrievalResult(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    stuffed_context: str


class RAGAnswer(BaseModel):
    query: str
    answer: str
    reasoning: str
    sources: List[str]  # source_names used in answer context
    retrieval: RetrievalResult