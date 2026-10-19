import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from schemas import DatasetRowChurn

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"

_loaded_rows: int | None = None


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    global _loaded_rows
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


def dataset_state(path: Path = DATASET_PATH) -> dict:
    # состояние опрашивает health-чек, поэтому ни файл, ни сам get_dataset не трогаем:
    # число строк запомнено при загрузке
    cached = get_dataset.cache_info().currsize > 0
    return {
        "available": path.exists(),
        "loaded": cached,
        "n_rows": _loaded_rows if cached else None,
    }
