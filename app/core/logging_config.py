import logging
import os

from app.core.config import DEFAULT_LOG_LEVEL, LOG_FORMAT


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    # force=True перебивает настройку, которую мог оставить uvicorn или импортёр
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
