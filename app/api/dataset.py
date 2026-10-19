from fastapi import APIRouter, Query

from app.api.deps import Dataset, build_balance
from app.api.docs import (
    DATASET_LOAD_RESPONSE,
    DATASET_NOT_FOUND_RESPONSE,
    EMPTY_DATASET_RESPONSE,
    PREVIEW_ERROR_RESPONSE,
    SPLIT_ERROR_RESPONSE,
)
from app.core.config import RANDOM_STATE, TEST_SIZE
from app.core.errors import DATA_PREPARATION_FAILED, EMPTY_DATASET, ServiceError
from app.ml.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_data,
    split_train_test,
)
from app.schemas import DatasetInfo, DatasetRowChurn, SplitInfo

router = APIRouter(prefix="/dataset")


@router.get(
    "/preview",
    response_model=list[DatasetRowChurn],
    responses={
        404: DATASET_NOT_FOUND_RESPONSE,
        422: PREVIEW_ERROR_RESPONSE,
    },
)
def dataset_preview(df: Dataset, n: int = Query(5, ge=1, le=100)):
    return df.head(n).to_dict(orient="records")


@router.get(
    "/info",
    response_model=DatasetInfo,
    responses={
        400: EMPTY_DATASET_RESPONSE,
        404: DATASET_NOT_FOUND_RESPONSE,
        422: DATASET_LOAD_RESPONSE,
    },
)
def dataset_info(df: Dataset):
    if df.empty:
        raise ServiceError(400, EMPTY_DATASET, "dataset is empty")

    balance = build_balance(df["churn"])
    return DatasetInfo(
        n_rows=len(df),
        n_columns=len(df.columns),
        features=list(df.columns),
        churn_distribution=balance.churn_distribution,
        churn_share=balance.churn_share,
    )


@router.get(
    "/split-info",
    response_model=SplitInfo,
    responses={
        400: EMPTY_DATASET_RESPONSE,
        404: DATASET_NOT_FOUND_RESPONSE,
        422: SPLIT_ERROR_RESPONSE,
    },
)
def dataset_split_info(df: Dataset):
    if df.empty:
        raise ServiceError(400, EMPTY_DATASET, "dataset is empty")

    try:
        X, y = prepare_data(df)
        _, _, y_train, y_test = split_train_test(X, y)
    except (KeyError, ValueError) as error:
        raise ServiceError(
            422, DATA_PREPARATION_FAILED, f"could not prepare dataset: {error}"
        )

    return SplitInfo(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratified=True,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        train=build_balance(y_train),
        test=build_balance(y_test),
    )
