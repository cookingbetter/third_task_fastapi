from functools import lru_cache
from pathlib import Path

import pandas as pd

from schemas import DatasetRowChurn

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows = [DatasetRowChurn(**record) for record in raw.to_dict(orient="records")]
    return pd.DataFrame([row.model_dump() for row in rows])


@lru_cache(maxsize=1)
def get_dataset() -> pd.DataFrame:
    return load_dataset()
