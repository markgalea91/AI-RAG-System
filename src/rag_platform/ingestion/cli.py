import argparse
import json
from dataclasses import asdict

from rag_platform.config.settings import get_settings
from rag_platform.common.logging import setup_logging
from rag_platform.ingestion.service import IngestionService


def main():
    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument("--data-dir", default="data/mtca_output/mtca_output_mt", help="Where to read data from")
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", action="store_false", dest="recursive")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    parser.add_argument("--recreate", action="store_true", default=False)
    parser.add_argument("--report-dir", default="runs/ingestion", help="Where to write reports/artifacts")
    args = parser.parse_args()

    settings = get_settings()
    logger = setup_logging()

    service = IngestionService(settings=settings, logger=logger)
    report = service.ingest_directory(
        directory=args.data_dir,
        recursive=args.recursive,
        overwrite=args.overwrite,
        recreate=args.recreate,
        report_dir=args.report_dir
    )

    print(json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()