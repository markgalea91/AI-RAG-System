import os
from email.policy import default
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
from triton.knobs import env


class Settings(BaseSettings):
    """
    Central configuration for the RAG platform.
    Values are loaded from environment variables when available.
    """

    # ------------------------
    # Core Vector DB Settings
    # ------------------------
    milvus_uri: str = Field(default="http://localhost:19530", env="MILVUS_URI")
    collection_name: str = Field(default="MTCA_ENG_HYB", env="COLLECTION_NAME")
    sparse_retrieval: bool = Field(default=True, env="SPARSE_RETRIEVAL")
    top_k: int = Field(default=3, env="TOP_K")
    max_chars_per_chunk: int = Field(default=1200, env="MAX_CHAR_PER_CHUNK")
    dense_dim: int = Field(default=2048, env="DENSE_DIM")
    gpu_cagra: bool = Field(default=True, env="GPU_CAGRA")
    hybrid_search: bool = Field(default=False, env="HYBRID_SEARCH")

    # ------------------------
    # nv-ingest Client
    # ------------------------
    nv_ingest_host: str = Field(default="localhost", env="NV_INGEST_HOST")
    nv_ingest_port: int = Field(default=7670, env="NV_INGEST_PORT")

    # ------------------------
    # Embeddings
    # ------------------------
    embedding_endpoint: str = Field(default="http://localhost:8012/v1", env="EMBEDDING_NIM_ENDPOINT")
    embedding_model: str = Field(default="nvidia/llama-3.2-nv-embedqa-1b-v2", env="EMBEDDING_NIM_MODEL_NAME")

    # ------------------------
    # LLM Models
    # ------------------------
    translator_model: str = Field(default="translategemma:27b", env="TRAN_MODEL")
    generation_model: str = Field(default="nemotron-3-nano:30b", env="GEN_MODEL")
    evaluation_model: str = Field(default="qwen2.5:7b-instruct", env="EVAL_MODEL")
    reasoning: bool = Field(default=True, env="REASONING")
    # ------------------------
    # Ingestion Settings
    # ------------------------
    batch_size_normal: int = Field(default=5, env="BATCH_SIZE_NORMAL")
    batch_size_heavy: int = Field(default=1, env="BATCH_SIZE_HEAVY")
    heavy_size_mb: float = Field(default=50.0, env="HEAVY_SIZE_MB")
    heavy_pdf_pages: int = Field(default=80, env="HEAVY_PDF_PAGES")
    sleep_seconds: int = Field(default=2, env="INGEST_SLEEP_SECONDS")
    data_dir: str = Field(default="data", env="DATA_DIR")
    quarantine_dir: str = Field(default="data/quarantine", env="QUARANTINE_DIR")

    # ------------------------
    # Evaluation Runtime
    # ------------------------
    ragas_max_workers: int = Field(default=1, env="RAGAS_MAX_WORKERS")
    ragas_timeout: int = Field(default=300, env="RAGAS_TIMEOUT")

    # ------------------------
    # .NET LLM API
    # ------------------------
    use_api: bool = Field(default=True, env="USE_LLM_API")
    is_chat: bool = Field(default=False, env="IS_CHAT")
    base_url: str = Field(default="https://llmapi.2iltd.com", env="BASE_URL")
    provider: str = Field(default="Ollama", env="PROVIDER")
    api_client_guid: str = Field(default="7fe78163-0c8f-4bd1-b36b-bc75843bb69f", env="CLIENT_GUID")


    # ----------------
    # LOGGING
    # ----------------
    RAG_LOGGING_ENABLED: bool = Field(default=True, env="RAG_LOGGING_ENABLED")
    RAG_LOG_DB_PATH: str = Field(default="./src/rag_platform/logging/logs/rag_logs.sqlite3", env="RAG_LOG_DB_PATH")

    # Safety: cap chunk text stored in logs (avoid huge DB rows)
    LOG_CHUNK_TEXT_MAX: int = Field(default=1200, env="RAG_LOG_CHUNK_TEXT_MAX")
    # Safety: cap response length stored
    LOG_RESPONSE_TEXT_MAX: int = Field(default=20000, env="RAG_LOG_RESPONSE_TEXT_MAX")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings object.
    Use this everywhere instead of instantiating Settings() manually.
    """
    settings = Settings()

    # 🔥 IMPORTANT: force env vars BEFORE nv_ingest is used
    os.environ["EMBEDDING_NIM_ENDPOINT"] = settings.embedding_endpoint
    os.environ["EMBEDDING_NIM_MODEL_NAME"] = settings.embedding_model

    return Settings()