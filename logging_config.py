import logging
import os

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DEFAULT_LEVEL = "INFO"


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", DEFAULT_LEVEL).upper()
    # force=True перебивает настройку, которую мог оставить uvicorn или импортёр
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
