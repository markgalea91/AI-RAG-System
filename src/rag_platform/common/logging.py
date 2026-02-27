import logging
import os
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    *,
    name: str = "rag_platform",
) -> logging.Logger:
    """
    Configure and return a logger for the project.
    Call once at startup in each CLI/app.
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Keep third-party logs quieter unless you want them
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger