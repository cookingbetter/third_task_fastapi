from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from dataset import get_dataset
from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    class_distribution,
    prepare_data,
    split_train_test,
)
from schemas import (
    ClassBalance,
    DatasetInfo,
    DatasetRowChurn,
    FeatureVectorChurn,
    SplitInfo,
)

app = FastAPI(title="ML Churn Service")


def dataset_dependency() -> pd.DataFrame:
    try:
        return get_dataset()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="dataset file not found")


Dataset = Annotated[pd.DataFrame, Depends(dataset_dependency)]


def build_balance(y: pd.Series) -> ClassBalance:
    counts = class_distribution(y)
    return ClassBalance(
        n_rows=len(y),
        churn_distribution=counts,
        churn_share={k: round(v / len(y), 4) for k, v in counts.items()},
    )


@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@app.post("/predict", response_model=FeatureVectorChurn)
def predict(features: FeatureVectorChurn):
    return features


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def dataset_preview(df: Dataset, n: int = Query(5, ge=1, le=100)):
    return df.head(n).to_dict(orient="records")


@app.get("/dataset/info", response_model=DatasetInfo)
def dataset_info(df: Dataset):
    balance = build_balance(df["churn"])
    return DatasetInfo(
        n_rows=len(df),
        n_columns=len(df.columns),
        features=list(df.columns),
        churn_distribution=balance.churn_distribution,
        churn_share=balance.churn_share,
    )


@app.get("/dataset/split-info", response_model=SplitInfo)
def dataset_split_info(df: Dataset):
    X, y = prepare_data(df)
    _, _, y_train, y_test = split_train_test(X, y)
    return SplitInfo(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratified=True,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        train=build_balance(y_train),
        test=build_balance(y_test),
    )
