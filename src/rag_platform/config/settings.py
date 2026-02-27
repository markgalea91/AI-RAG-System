import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Central configuration for the RAG platform.
    Values are loaded from environment variables when available.
    """

    # ------------------------
    # Core Vector DB Settings
    # ------------------------
    milvus_uri: str = Field(default="http://localhost:19530", env="MILVUS_URI")
    collection_name: str = Field(default="MTCA_MT_HYB", env="COLLECTION_NAME")
    sparse_retrieval: bool = Field(default=True, env="SPARSE_RETRIEVAL")
    top_k: int = Field(default=3, env="TOP_K")
    dense_dim: int = Field(default=2048, env="DENSE_DIM")
    gpu_cagra: bool = Field(default=True, env="GPU_CAGRA")
    hybrid_search: bool = Field(default=True, env="HYBRID_SEARCH")

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
    translator_model: str = Field(default="translategemma:12b", env="TRAN_MODEL")
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