import json
import datetime as dt
import sqlite3
from typing import List, Dict, Optional
from rag_platform.config.settings import get_settings
from pathlib import Path


s = get_settings()

RAG_LOGGING_ENABLED = s.RAG_LOGGING_ENABLED
RAG_LOG_DB_PATH = s.RAG_LOG_DB_PATH
# Safety: cap chunk text stored in logs (avoid huge DB rows)
LOG_CHUNK_TEXT_MAX = s.LOG_CHUNK_TEXT_MAX
# Safety: cap response length stored
LOG_RESPONSE_TEXT_MAX = s.LOG_RESPONSE_TEXT_MAX


class BaseAuditLogger:
    def start(self, trace_id: str, query: Optional[str]):
        pass

    def log_retrieval(self, trace_id: str, chunks: List[Dict]):
        pass

    def finish(
        self,
        trace_id: str,
        response_text: Optional[str],
        reasoning: Optional[str],
        duration_ms: int,
        status: str,
        error: Optional[str] = None,
    ):
        pass


class NoAuditLogger(BaseAuditLogger):
    pass


class SQLiteAuditLogger(BaseAuditLogger):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        db_path = Path(self.db_path).expanduser()

        # If relative path, make it absolute based on *project root* or current file
        if not db_path.is_absolute():
            # This resolves relative paths from the repository root:
            # rag_audit.py is at: src/rag_platform/logging/rag_audit.py
            # parents[3] => project root (…/2i AI RAG System)
            project_root = Path(__file__).resolve().parents[3]
            db_path = (project_root / db_path).resolve()

        # ✅ Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # store normalized absolute path back (helps debugging)
        self.db_path = str(db_path)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_request_logs (
                    trace_id TEXT PRIMARY KEY,
                    created_utc TEXT NOT NULL,
                    duration_ms INTEGER,
                    query TEXT,
                    chunks_json TEXT,
                    response_text TEXT,
                    reasoning TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                );
                """
            )
            conn.commit()

    def start(self, trace_id: str, query: Optional[str]):
        created_utc = dt.datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO rag_request_logs(trace_id, created_utc, query, status)
                VALUES (?, ?, ?, ?)
                """,
                (trace_id, created_utc, query, "started"),
            )
            conn.commit()

    def log_retrieval(self, trace_id: str, chunks: List[Dict]):
        capped_chunks = []
        for c in chunks:
            text = c.get("text")
            if isinstance(text, str) and len(text) > LOG_CHUNK_TEXT_MAX:
                text = text[:LOG_CHUNK_TEXT_MAX] + "…"

            capped_chunks.append(
                {
                    "text": text,
                    "source_name": c.get("source_name"),
                    "page_number": c.get("page_number"),
                    "score": c.get("score"),
                }
            )

        chunks_json = json.dumps(capped_chunks, ensure_ascii=False)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE rag_request_logs
                SET chunks_json = ?
                WHERE trace_id = ?
                """,
                (chunks_json, trace_id),
            )
            conn.commit()

    def finish(
        self,
        trace_id: str,
        response_text: Optional[str],
        reasoning: Optional[str],
        duration_ms: int,
        status: str,
        error: Optional[str] = None,
    ):
        if isinstance(response_text, str) and len(response_text) > LOG_RESPONSE_TEXT_MAX:
            response_text = response_text[:LOG_RESPONSE_TEXT_MAX] + "…"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE rag_request_logs
                SET response_text = ?, reasoning = ?, duration_ms = ?, status = ?, error = ?
                WHERE trace_id = ?
                """,
                (response_text, reasoning, duration_ms, status, error, trace_id),
            )
            conn.commit()


def build_audit_logger() -> BaseAuditLogger:
    if not RAG_LOGGING_ENABLED:
        return NoAuditLogger()
    return SQLiteAuditLogger(RAG_LOG_DB_PATH)


# Global singleton logger
audit_logger = build_audit_logger()