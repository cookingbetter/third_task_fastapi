from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from dataset import get_dataset
from model import (
    ModelArtifact,
    get_artifact,
    load_churn_model,
    save_churn_model,
    set_artifact,
    train_churn_model,
)
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
    ModelStatus,
    SplitInfo,
    TrainMetrics,
    TrainResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = load_churn_model()
    if artifact is not None:
        set_artifact(artifact)
    yield


app = FastAPI(title="ML Churn Service", lifespan=lifespan)


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


@app.post("/model/train", response_model=TrainResponse)
def model_train(df: Dataset):
    if df.empty:
        raise HTTPException(status_code=400, detail="dataset is empty")
    pipeline, metrics = train_churn_model(df)
    artifact = ModelArtifact(
        pipeline=pipeline,
        trained_at=datetime.now(timezone.utc),
        metrics=metrics,
    )
    save_churn_model(artifact)
    set_artifact(artifact)
    return TrainResponse(
        status="trained",
        algorithm="LogisticRegression",
        n_train=metrics["n_train"],
        n_test=metrics["n_test"],
        metrics=TrainMetrics(accuracy=metrics["accuracy"], f1=metrics["f1"]),
    )


@app.get("/model/status", response_model=ModelStatus)
def model_status():
    artifact = get_artifact()
    if artifact is None:
        return ModelStatus(is_trained=False, trained_at=None, metrics=None)
    return ModelStatus(
        is_trained=True,
        trained_at=artifact.trained_at,
        metrics=TrainMetrics(
            accuracy=artifact.metrics["accuracy"],
            f1=artifact.metrics["f1"],
        ),
    )
