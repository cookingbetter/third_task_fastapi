import logging
import math
from typing import Annotated

import pandas as pd
from fastapi import Depends
from pydantic import ValidationError

from app.core.errors import (
    DATA_PREPARATION_FAILED,
    DATASET_NOT_FOUND,
    NON_FINITE_VALUE,
    ServiceError,
)
from app.ml.dataset import get_dataset
from app.ml.preprocessing import NUMERIC_FEATURES, class_distribution
from app.schemas import ClassBalance, FeatureVectorChurn, TrainingRecord

logger = logging.getLogger(__name__)


def dataset_dependency() -> pd.DataFrame:
    try:
        return get_dataset()
    except FileNotFoundError:
        raise ServiceError(404, DATASET_NOT_FOUND, "dataset file not found")
    except (KeyError, ValueError) as error:
        # в файле не те столбцы или значения, которых не ждёт DatasetRowChurn
        raise ServiceError(
            422,
            DATA_PREPARATION_FAILED,
            f"could not load dataset: {(str(error).splitlines() or ['unknown error'])[0]}",
        )


Dataset = Annotated[pd.DataFrame, Depends(dataset_dependency)]


def build_balance(y: pd.Series) -> ClassBalance:
    counts = class_distribution(y)
    return ClassBalance(
        n_rows=len(y),
        churn_distribution=counts,
        churn_share={k: round(v / len(y), 4) for k, v in counts.items()},
    )


def valid_records(raw: list) -> list[TrainingRecord]:
    records = []
    for position, item in enumerate(raw):
        try:
            records.append(TrainingRecord(**item))
        except (TypeError, ValidationError):
            # файл истории могли поправить руками, одна запись не повод падать
            logger.warning("skipping malformed history record at %s", position)
    return records


def ensure_finite(objects: list[FeatureVectorChurn]) -> None:
    for position, obj in enumerate(objects):
        for field in NUMERIC_FEATURES:
            value = getattr(obj, field)
            try:
                is_finite = math.isfinite(value)
            except OverflowError:
                # целые в python неограничены, а признаки уходят в модель как float
                is_finite = False
            if not is_finite:
                raise ServiceError(
                    422,
                    NON_FINITE_VALUE,
                    f"{field} must be a finite number",
                    {"position": position, "field": field},
                )
