# import os
# import json
# import time
# import uuid
# import datetime as dt
# from typing import Any, Dict, List, Optional
#
# import aiosqlite
# import httpx
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from starlette.middleware.base import BaseHTTPMiddleware
# from rag_platform.config.settings import get_settings
#
#
# # -----------------------------
# # Config
# # -----------------------------
# s = get_settings()
#
# RAG_LOGGING_ENABLED = s.RAG_LOGGING_ENABLED
# RAG_LOG_DB_PATH = s.RAG_LOG_DB_PATH
# # Safety: cap chunk text stored in logs (avoid huge DB rows)
# LOG_CHUNK_TEXT_MAX = s.LOG_CHUNK_TEXT_MAX
# # Safety: cap response length stored
# LOG_RESPONSE_TEXT_MAX = s.LOG_RESPONSE_TEXT_MAX
#
#
# # -----------------------------
# # Logging Interfaces
# # -----------------------------
# class RagAuditLogger:
#     async def init(self) -> None:
#         raise NotImplementedError
#
#     async def start(self, trace_id: str, query: Optional[str]) -> None:
#         raise NotImplementedError
#
#     async def log_retrieval(self, trace_id: str, chunks: List[Dict[str, Any]]) -> None:
#         raise NotImplementedError
#
#     async def finish(
#         self,
#         trace_id: str,
#         response_text: Optional[str],
#         duration_ms: int,
#         status: str,
#         error: Optional[str] = None,
#     ) -> None:
#         raise NotImplementedError
#
#
# class NoopAuditLogger(RagAuditLogger):
#     async def init(self) -> None:
#         return
#
#     async def start(self, trace_id: str, query: Optional[str]) -> None:
#         return
#
#     async def log_retrieval(self, trace_id: str, chunks: List[Dict[str, Any]]) -> None:
#         return
#
#     async def finish(
#         self,
#         trace_id: str,
#         response_text: Optional[str],
#         duration_ms: int,
#         status: str,
#         error: Optional[str] = None,
#     ) -> None:
#         return
#
#
# class SQLiteAuditLogger(RagAuditLogger):
#     def __init__(self, db_path: str):
#         self.db_path = db_path
#
#     async def init(self) -> None:
#         async with aiosqlite.connect(self.db_path) as db:
#             await db.execute(
#                 """
#                 CREATE TABLE IF NOT EXISTS rag_request_logs (
#                   trace_id TEXT PRIMARY KEY,
#                   created_utc TEXT NOT NULL,
#                   duration_ms INTEGER,
#                   query TEXT,
#                   chunks_json TEXT,
#                   response_text TEXT,
#                   status TEXT NOT NULL,
#                   error TEXT
#                 );
#                 """
#             )
#             # Useful indexes for quick filtering
#             await db.execute("CREATE INDEX IF NOT EXISTS idx_rag_logs_created ON rag_request_logs(created_utc);")
#             await db.execute("CREATE INDEX IF NOT EXISTS idx_rag_logs_status ON rag_request_logs(status);")
#             await db.commit()
#
#     async def start(self, trace_id: str, query: Optional[str]) -> None:
#         created_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
#         async with aiosqlite.connect(self.db_path) as db:
#             await db.execute(
#                 """
#                 INSERT INTO rag_request_logs(trace_id, created_utc, query, status)
#                 VALUES (?, ?, ?, ?)
#                 """,
#                 (trace_id, created_utc, query, "started"),
#             )
#             await db.commit()
#
#     async def log_retrieval(self, trace_id: str, chunks: List[Dict[str, Any]]) -> None:
#         # Apply caps to avoid huge DB rows
#         capped_chunks = []
#         for c in chunks:
#             text = c.get("text")
#             if isinstance(text, str) and len(text) > LOG_CHUNK_TEXT_MAX:
#                 text = text[:LOG_CHUNK_TEXT_MAX] + "…"
#             capped_chunks.append(
#                 {
#                     "text": text,
#                     "source_name": c.get("source_name"),
#                     "page_number": c.get("page_number"),
#                     "score": c.get("score"),
#                 }
#             )
#
#         chunks_json = json.dumps(capped_chunks, ensure_ascii=False)
#         async with aiosqlite.connect(self.db_path) as db:
#             await db.execute(
#                 """
#                 UPDATE rag_request_logs
#                 SET chunks_json = ?
#                 WHERE trace_id = ?
#                 """,
#                 (chunks_json, trace_id),
#             )
#             await db.commit()
#
#     async def finish(
#         self,
#         trace_id: str,
#         response_text: Optional[str],
#         duration_ms: int,
#         status: str,
#         error: Optional[str] = None,
#     ) -> None:
#         if isinstance(response_text, str) and len(response_text) > LOG_RESPONSE_TEXT_MAX:
#             response_text = response_text[:LOG_RESPONSE_TEXT_MAX] + "…"
#
#         async with aiosqlite.connect(self.db_path) as db:
#             await db.execute(
#                 """
#                 UPDATE rag_request_logs
#                 SET response_text = ?, duration_ms = ?, status = ?, error = ?
#                 WHERE trace_id = ?
#                 """,
#                 (response_text, duration_ms, status, error, trace_id),
#             )
#             await db.commit()
#
#
# def build_audit_logger() -> RagAuditLogger:
#     if not RAG_LOGGING_ENABLED:
#         return NoopAuditLogger()
#     return SQLiteAuditLogger(RAG_LOG_DB_PATH)
#
#
# # -----------------------------
# # Middleware: trace_id + timing + finalize
# # -----------------------------
# class TraceMiddleware(BaseHTTPMiddleware):
#     def __init__(self, app: FastAPI, audit_logger: RagAuditLogger):
#         super().__init__(app)
#         self.audit_logger = audit_logger
#
#     async def dispatch(self, request: Request, call_next):
#         trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
#         request.state.trace_id = trace_id
#
#         start = time.perf_counter()
#         try:
#             response = await call_next(request)
#             duration_ms = int((time.perf_counter() - start) * 1000)
#             # If your handler already logged finish, you can skip this.
#             # But it's safer to only finalize in handler where response_text is known.
#             response.headers["X-Trace-Id"] = trace_id
#             return response
#         except Exception as e:
#             duration_ms = int((time.perf_counter() - start) * 1000)
#             await self.audit_logger.finish(
#                 trace_id=trace_id,
#                 response_text=None,
#                 duration_ms=duration_ms,
#                 status="failed",
#                 error=repr(e),
#             )
#             return JSONResponse(
#                 status_code=500,
#                 content={"error": "Internal Server Error", "trace_id": trace_id},
#                 headers={"X-Trace-Id": trace_id},
#             )
#
#
# # -----------------------------
# # Example FastAPI app
# # -----------------------------
# app = FastAPI()
# audit_logger = build_audit_logger()
#
#
# @app.on_event("startup")
# async def _startup():
#     await audit_logger.init()
#     app.add_middleware(TraceMiddleware, audit_logger=audit_logger)
#
#
# # -----------------------------
# # Example: your RAG endpoint skeleton
# # -----------------------------
# @app.post("/rag")
# async def rag_endpoint(payload: Dict[str, Any], request: Request):
#     """
#     Expected payload example:
#       { "query": "....", "top_k": 5, ... }
#     """
#     trace_id: str = request.state.trace_id
#     query: Optional[str] = payload.get("query")
#
#     await audit_logger.start(trace_id=trace_id, query=query)
#
#     t0 = time.perf_counter()
#     try:
#         # 1) Retrieval (replace with your actual retrieval)
#         retrieved_chunks = await retrieve_chunks(query=query, top_k=int(payload.get("top_k", 5)))
#         await audit_logger.log_retrieval(trace_id=trace_id, chunks=retrieved_chunks)
#
#         # 2) Call .NET LLM API (pass trace_id for correlation)
#         llm_response_text = await call_llm_dotnet_api(
#             trace_id=trace_id,
#             query=query,
#             chunks=retrieved_chunks,
#         )
#
#         duration_ms = int((time.perf_counter() - t0) * 1000)
#
#         await audit_logger.finish(
#             trace_id=trace_id,
#             response_text=llm_response_text,
#             duration_ms=duration_ms,
#             status="success",
#             error=None,
#         )
#
#         return {"trace_id": trace_id, "answer": llm_response_text}
#
#     except Exception as e:
#         duration_ms = int((time.perf_counter() - t0) * 1000)
#         await audit_logger.finish(
#             trace_id=trace_id,
#             response_text=None,
#             duration_ms=duration_ms,
#             status="failed",
#             error=repr(e),
#         )
#         return JSONResponse(
#             status_code=500,
#             content={"trace_id": trace_id, "error": "RAG failed"},
#         )
#
#
# # -----------------------------
# # Replace these stubs with your code
# # -----------------------------
# async def retrieve_chunks(query: Optional[str], top_k: int) -> List[Dict[str, Any]]:
#     # TODO: integrate your Milvus retrieval here
#     # Return list of dicts with keys: text, source_name, page_number, score
#     if not query:
#         return []
#     return [
#         {
#             "text": "Example chunk text ...",
#             "source_name": "example.pdf",
#             "page_number": 1,
#             "score": 0.87,
#         }
#     ]
#
#
# async def call_llm_dotnet_api(trace_id: str, query: Optional[str], chunks: List[Dict[str, Any]]) -> str:
#     # Build payload to your .NET LLM integration API
#     dotnet_payload = {
#         "query": query,
#         "chunks": chunks,  # or your structured model
#     }
#
#     # IMPORTANT: pass trace_id to .NET for correlation
#     headers = {"X-Trace-Id": trace_id}
#
#     # Replace with your actual .NET URL
#     dotnet_url = os.getenv("DOTNET_LLM_URL", "http://localhost:5000/llm/answer")
#
#     async with httpx.AsyncClient(timeout=60.0) as client:
#         r = await client.post(dotnet_url, json=dotnet_payload, headers=headers)
#         r.raise_for_status()
#
#         data = r.json()
#         # adapt to your .NET response schema:
#         # e.g. { "answer": "...", "sources": [...] }
#         return data.get("answer") or json.dumps(data)

import os
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