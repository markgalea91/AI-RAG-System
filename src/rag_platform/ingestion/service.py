from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from nv_ingest_client.client import Ingestor, NvIngestClient

from rag_platform.config.settings import Settings
from rag_platform.common.logging import setup_logging
from rag_platform.common.types import IngestionFailure, IngestionReport
from rag_platform.ingestion.json_adapter import load_json_records

from .file_prep import prepare_files
from .heavy_router import classify_files_heavy


def chunk_list(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class IngestionService:
    def __init__(self, settings: Settings, logger=None):
        self.settings = settings
        self.logger = logger or setup_logging()

        self.client = NvIngestClient(
            message_client_hostname=settings.nv_ingest_host,
            message_client_port=settings.nv_ingest_port,
        )

    def _build_ingestor(
        self,
        files: List[str],
        *,
        heavy_mode: bool,
        temp_output_dir: str,
        recreate: bool,
    ) -> Ingestor:
        s = self.settings

        extract_kwargs = dict(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=False,
            table_output_format="markdown",
            text_depth="page",
            extract_infographics=not heavy_mode,
        )

        return (
            Ingestor(client=self.client)
            .save_to_disk(output_directory=temp_output_dir, cleanup=True)
            .files(files)
            .extract(**extract_kwargs)
            .split(chunk_size=1000,
                   chunk_overlap=150)
            .embed()
            .vdb_upload(
                collection_name=s.collection_name,
                milvus_uri=s.milvus_uri,
                sparse=s.sparse_retrieval,
                dense_dim=s.dense_dim,
                recreate=recreate,
                hybrid=s.hybrid_search,
                gpu_cagra=s.gpu_cagra
            )
        )

    def ingest_directory(
        self,
        directory: Optional[str] = None,
        *,
        recursive: bool = True,
        overwrite: bool = True,
        recreate: bool = False,
        sleep_seconds: Optional[int] = None,
        report_dir: Optional[str] = None,
    ) -> IngestionReport:
        """
        Ingest a directory of files into Milvus via nv-ingest.

        report_dir (optional): if provided, saves:
          - report.json
          - failures.jsonl
          under report_dir/<run_id>/
        """
        s = self.settings
        data_dir = directory or s.data_dir
        sleep_seconds = s.sleep_seconds if sleep_seconds is None else sleep_seconds

        run_id = _utc_run_id()
        temp_output_dir = f"temp_ingest_results/{run_id}"

        t0 = time.time()

        # 1) Prepare files
        files = prepare_files(
            directory=data_dir,
            quarantine_dir=s.quarantine_dir,
            recursive=recursive,
            overwrite=overwrite,
        )

        # 2) Split heavy vs normal
        normal_files, heavy_files, heavy_reasons = classify_files_heavy(
            files,
            heavy_size_mb=s.heavy_size_mb,
            pdf_heavy_pages=s.heavy_pdf_pages,
        )

        self.logger.info(
            f"[ingest] run_id={run_id} total={len(files)} normal={len(normal_files)} heavy={len(heavy_files)} recreate={recreate}"
        )

        all_results: List[Dict[str, Any]] = []
        failures: List[IngestionFailure] = []

        # Helper to record failures
        def record_failure(file: str, reason: str, details: Dict[str, Any]):
            failures.append(IngestionFailure(file=file, reason=reason, details=details))

        # 3) Normal pass
        for batch_idx, batch in enumerate(chunk_list(normal_files, s.batch_size_normal), start=1):
            self.logger.info(f"[ingest][normal] batch={batch_idx} size={len(batch)}")
            start = time.time()

            try:
                ingestor = self._build_ingestor(
                    batch,
                    heavy_mode=False,
                    temp_output_dir=temp_output_dir,
                    recreate=recreate,
                )
                results, batch_failures = ingestor.ingest(show_progress=True, return_failures=True)
                if results:
                    all_results.extend(results)
                if batch_failures:
                    for bf in batch_failures:
                        record_failure(
                            file=str(bf.get("file", "")),
                            reason="nv_ingest_failure",
                            details=bf,
                        )
            except Exception as e:
                self.logger.exception(f"[ingest][normal] batch={batch_idx} crashed")
                record_failure(
                    file=";".join(batch[:10]),
                    reason="exception_normal_batch",
                    details={"error": repr(e), "batch": batch_idx, "files_count": len(batch)},
                )

            self.logger.info(f"[ingest][normal] batch={batch_idx} elapsed={time.time() - start:.2f}s")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        # 4) Heavy pass
        for batch_idx, batch in enumerate(chunk_list(heavy_files, s.batch_size_heavy), start=1):
            self.logger.info(f"[ingest][heavy] batch={batch_idx} size={len(batch)}")
            start = time.time()

            try:
                ingestor = self._build_ingestor(
                    batch,
                    heavy_mode=True,
                    temp_output_dir=temp_output_dir,
                    recreate=recreate,
                )
                results, batch_failures = ingestor.ingest(show_progress=True, return_failures=True)
                if results:
                    all_results.extend(results)
                if batch_failures:
                    for bf in batch_failures:
                        record_failure(
                            file=str(bf.get("file", "")),
                            reason="nv_ingest_failure",
                            details=bf,
                        )
            except Exception as e:
                self.logger.exception(f"[ingest][heavy] batch={batch_idx} crashed")
                record_failure(
                    file=";".join(batch[:10]),
                    reason="exception_heavy_batch",
                    details={"error": repr(e), "batch": batch_idx, "files_count": len(batch)},
                )

            self.logger.info(f"[ingest][heavy] batch={batch_idx} elapsed={time.time() - start:.2f}s")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        elapsed = time.time() - t0

        report = IngestionReport(
            total_files=len(files),
            normal_files=len(normal_files),
            heavy_files=len(heavy_files),
            results_count=len(all_results),
            failures=failures,
            elapsed_seconds=elapsed,
            heavy_reasons=heavy_reasons,
        )

        self.logger.info(
            f"[ingest] run_id={run_id} finished elapsed={elapsed:.2f}s results={report.results_count} failures={len(report.failures)}"
        )

        # 5) Persist report artifacts (optional)
        if report_dir:
            out_dir = Path(report_dir) / run_id
            out_dir.mkdir(parents=True, exist_ok=True)

            (out_dir / "report.json").write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            # failures.jsonl for easy grepping/streaming
            with (out_dir / "failures.jsonl").open("w", encoding="utf-8") as f:
                for fail in failures:
                    f.write(json.dumps(asdict(fail), ensure_ascii=False, default=str) + "\n")

            self.logger.info(f"[ingest] wrote report to {out_dir}")

        return report

    # def ingest_path(
    #         self,
    #         path: str | Path,
    #         *,
    #         overwrite: bool,
    #         recreate: bool,
    #         report_dir: str | Path | None = None,
    # ):
    #     p = Path(path)
    #
    #     # ✅ Route JSON through adapter + ingest_docs
    #     if p.suffix.lower() == ".json":
    #         items = load_json_records(str(p))
    #
    #         docs = []
    #         for it in items:
    #             text = (it.get("text") or "").strip()
    #             if not text:
    #                 continue
    #             docs.append({
    #                 "text": text,
    #                 "metadata": it.get("metadata") or {},
    #             })
    #
    #         if not docs:
    #             self.logger.warning("No usable JSON docs found in %s", p)
    #             return None
    #
    #         self.logger.info("Ingesting JSON via ingest_docs: %s (%d docs)", p, len(docs))
    #         return self.ingest_docs(
    #             docs,
    #             overwrite=overwrite,
    #             recreate=recreate,
    #             report_dir=report_dir,
    #             split_kwargs={"chunk_size": 1000, "chunk_overlap": 150},
    #         )
    #
    #     # ✅ Everything else follows your existing file path
    #     self.logger.info("Ingesting file via ingest_file: %s", p)
    #     return self.ingest_file(
    #         str(p),
    #         overwrite=overwrite,
    #         recreate=recreate,
    #         report_dir=report_dir,
    #     )


# from __future__ import annotations
#
# import json
# import time
# from dataclasses import asdict
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any, Dict, List, Optional
#
# from nv_ingest_client.client import Ingestor, NvIngestClient
#
# from rag_platform.config.settings import Settings
# from rag_platform.common.logging import setup_logging
# from rag_platform.common.types import IngestionFailure, IngestionReport
# from rag_platform.ingestion.json_adapter import load_json_records
#
# from .file_prep import prepare_files
# from .heavy_router import classify_files_heavy
#
#
# def chunk_list(items: List[str], batch_size: int):
#     for i in range(0, len(items), batch_size):
#         yield items[i : i + batch_size]
#
#
# def _utc_run_id() -> str:
#     return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
#
#
# class IngestionService:
#     def __init__(self, settings: Settings, logger=None):
#         self.settings = settings
#         self.logger = logger or setup_logging()
#
#         self.client = NvIngestClient(
#             message_client_hostname=settings.nv_ingest_host,
#             message_client_port=settings.nv_ingest_port,
#         )
#
#     # --------------------------------------------------------
#     # STANDARD FILE INGESTOR (PDF, DOCX, TXT, etc.)
#     # --------------------------------------------------------
#     def _build_file_ingestor(
#         self,
#         files: List[str],
#         *,
#         heavy_mode: bool,
#         temp_output_dir: str,
#         recreate: bool,
#     ) -> Ingestor:
#
#         s = self.settings
#
#         extract_kwargs = dict(
#             extract_text=True,
#             extract_tables=True,
#             extract_charts=True,
#             extract_images=False,
#             table_output_format="markdown",
#             text_depth="page",
#             extract_infographics=not heavy_mode,
#         )
#
#         return (
#             Ingestor(client=self.client)
#             .save_to_disk(output_directory=temp_output_dir, cleanup=True)
#             .files(files)
#             .extract(**extract_kwargs)
#             .split(chunk_size=1000, chunk_overlap=150)
#             .embed()
#             .vdb_upload(
#                 collection_name=s.collection_name,
#                 milvus_uri=s.milvus_uri,
#                 sparse=s.sparse_retrieval,
#                 dense_dim=s.dense_dim,
#                 recreate=recreate,
#             )
#         )
#
#     # --------------------------------------------------------
#     # JSON STRUCTURED INGESTOR (uses .load(), NOT .extract())
#     # --------------------------------------------------------
#     def _build_json_ingestor(
#         self,
#         structured_file: str,
#         *,
#         temp_output_dir: str,
#         recreate: bool,
#     ) -> Ingestor:
#
#         s = self.settings
#
#         return (
#             Ingestor(client=self.client)
#             .save_to_disk(output_directory=temp_output_dir, cleanup=True)
#             .files([structured_file])   # ✅ KEY DIFFERENCE
#             .split(chunk_size=1000, chunk_overlap=150)
#             .embed()
#             .vdb_upload(
#                 collection_name=s.collection_name,
#                 milvus_uri=s.milvus_uri,
#                 sparse=s.sparse_retrieval,
#                 dense_dim=s.dense_dim,
#                 recreate=recreate
#             )
#         )
#
#     # --------------------------------------------------------
#     # MAIN DIRECTORY INGEST
#     # --------------------------------------------------------
#     def ingest_directory(
#         self,
#         directory: Optional[str] = None,
#         *,
#         recursive: bool = True,
#         overwrite: bool = True,
#         recreate: bool = False,
#         sleep_seconds: Optional[int] = None,
#         report_dir: Optional[str] = None,
#     ) -> IngestionReport:
#
#         s = self.settings
#         data_dir = directory or s.data_dir
#         sleep_seconds = s.sleep_seconds if sleep_seconds is None else sleep_seconds
#
#         run_id = _utc_run_id()
#         temp_output_dir = f"temp_ingest_results/{run_id}"
#
#         t0 = time.time()
#
#         files = prepare_files(
#             directory=data_dir,
#             quarantine_dir=s.quarantine_dir,
#             recursive=recursive,
#             overwrite=overwrite,
#         )
#
#         # 🔹 Split JSON vs other files
#         json_files = [f for f in files if Path(f).suffix.lower() == ".json"]
#         other_files = [f for f in files if Path(f).suffix.lower() != ".json"]
#
#         normal_files, heavy_files, heavy_reasons = classify_files_heavy(
#             other_files,
#             heavy_size_mb=s.heavy_size_mb,
#             pdf_heavy_pages=s.heavy_pdf_pages,
#         )
#
#         all_results: List[Dict[str, Any]] = []
#         failures: List[IngestionFailure] = []
#
#         def record_failure(file: str, reason: str, details: Dict[str, Any]):
#             failures.append(IngestionFailure(file=file, reason=reason, details=details))
#
#         # --------------------------------------------------------
#         # 1️⃣ JSON FILES (STRUCTURED LOAD)
#         # --------------------------------------------------------
#         for jf in json_files:
#             try:
#                 self.logger.info(f"[ingest][json] processing {jf}")
#
#                 records = load_json_records(jf)
#
#                 # Build structured nv-ingest format
#                 structured_payload = {"PageDataList": []}
#
#                 for rec in records:
#                     structured_payload["PageDataList"].append(
#                         {
#                             "entity": {
#                                 "text": rec["text"],
#                                 "source": {
#                                     "source_name": Path(jf).name,
#                                     "source_type": "json",
#                                 },
#                                 "content_metadata": {
#                                     "page_number": -1,
#                                     "language": rec["metadata"].get("language", "unknown"),
#                                     "uri": rec["metadata"].get("uri"),
#                                 },
#                             },
#                             "custom_content": rec["metadata"],  # keep small metadata only
#                         }
#                     )
#
#                 # Write temp structured file
#                 structured_path = f"{temp_output_dir}_{Path(jf).stem}_structured.json"
#                 Path(structured_path).parent.mkdir(parents=True, exist_ok=True)
#
#                 with open(structured_path, "w", encoding="utf-8") as f:
#                     json.dump(structured_payload, f, ensure_ascii=False)
#
#                 ingestor = self._build_json_ingestor(
#                     structured_path,
#                     temp_output_dir=temp_output_dir,
#                     recreate=recreate,
#                 )
#
#                 results, batch_failures = ingestor.ingest(
#                     show_progress=True, return_failures=True
#                 )
#
#                 if results:
#                     all_results.extend(results)
#
#                 if batch_failures:
#                     for bf in batch_failures:
#                         record_failure(
#                             file=str(bf.get("file", "")),
#                             reason="nv_ingest_failure",
#                             details=bf,
#                         )
#
#             except Exception as e:
#                 self.logger.exception(f"[ingest][json] failed file={jf}")
#                 record_failure(
#                     file=str(jf),
#                     reason="exception_json_file",
#                     details={"error": repr(e)},
#                 )
#
#         # --------------------------------------------------------
#         # 2️⃣ NORMAL FILES
#         # --------------------------------------------------------
#         for batch in chunk_list(normal_files, s.batch_size_normal):
#             try:
#                 ingestor = self._build_file_ingestor(
#                     batch,
#                     heavy_mode=False,
#                     temp_output_dir=temp_output_dir,
#                     recreate=recreate,
#                 )
#                 results, batch_failures = ingestor.ingest(
#                     show_progress=True, return_failures=True
#                 )
#                 if results:
#                     all_results.extend(results)
#             except Exception as e:
#                 record_failure(";".join(batch), "exception_normal_batch", {"error": repr(e)})
#
#         # --------------------------------------------------------
#         # 3️⃣ HEAVY FILES
#         # --------------------------------------------------------
#         for batch in chunk_list(heavy_files, s.batch_size_heavy):
#             try:
#                 ingestor = self._build_file_ingestor(
#                     batch,
#                     heavy_mode=True,
#                     temp_output_dir=temp_output_dir,
#                     recreate=recreate,
#                 )
#                 results, batch_failures = ingestor.ingest(
#                     show_progress=True, return_failures=True
#                 )
#                 if results:
#                     all_results.extend(results)
#             except Exception as e:
#                 record_failure(";".join(batch), "exception_heavy_batch", {"error": repr(e)})
#
#         elapsed = time.time() - t0
#
#         report = IngestionReport(
#             total_files=len(files),
#             normal_files=len(normal_files),
#             heavy_files=len(heavy_files),
#             results_count=len(all_results),
#             failures=failures,
#             elapsed_seconds=elapsed,
#             heavy_reasons=heavy_reasons,
#         )
#
#         return report