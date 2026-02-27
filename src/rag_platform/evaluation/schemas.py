# src/rag_platform/evaluation/schemas.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TestsetRow(BaseModel):
    user_input: str
    reference: str


class PipelineOutputRow(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truths: List[str]


class RagasSingleTurnRow(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: List[str]
    reference: str