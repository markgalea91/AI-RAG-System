from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Citation:
    source_name: str
    page_number: Optional[int] = None
    snippet: str = ""


@dataclass
class RAGResponse:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionFailure:
    file: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionReport:
    total_files: int
    normal_files: int
    heavy_files: int
    results_count: int
    failures: List[IngestionFailure] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    heavy_reasons: List[Dict[str, Any]] = field(default_factory=list)