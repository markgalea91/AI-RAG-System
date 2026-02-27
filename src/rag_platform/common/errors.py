class RAGPlatformError(Exception):
    """Base exception for rag_platform."""


class IngestionError(RAGPlatformError):
    pass


class RetrievalError(RAGPlatformError):
    pass


class EvaluationError(RAGPlatformError):
    pass