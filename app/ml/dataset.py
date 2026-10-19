import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.core.config import DATASET_PATH
from app.schemas import DatasetRowChurn

logger = logging.getLogger(__name__)

_loaded_rows: int | None = None


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    # путь читается при вызове, а не фиксируется значением по умолчанию:
    # иначе подменить его в тестах можно было бы только заменой самой функции
    global _loaded_rows
    path = path or DATASET_PATH
    raw = pd.read_csv(path)
    rows = [DatasetRowChurn(**record) for record in raw.to_dict(orient="records")]
    df = pd.DataFrame([row.model_dump() for row in rows])
    _loaded_rows = len(df)
    logger.info(
        "dataset loaded from %s: %s rows, %s columns", path, len(df), len(df.columns)
    )
    return df


@lru_cache(maxsize=1)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


def dataset_state(path: Path | None = None) -> dict:
    # состояние опрашивает health-чек, поэтому ни файл, ни сам get_dataset не трогаем:
    # число строк запомнено при загрузке
    path = path or DATASET_PATH
    cached = get_dataset.cache_info().currsize > 0
    return {
        "available": path.exists(),
        "loaded": cached,
        "n_rows": _loaded_rows if cached else None,
    }
