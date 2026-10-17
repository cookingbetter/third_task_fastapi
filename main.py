import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dataset import get_dataset
from model import (
    InvalidTrainingConfig,
    ModelArtifact,
    get_artifact,
    get_model,
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
    PredictionChurn,
    PredictionResponseChurn,
    SplitInfo,
    TrainingConfigChurn,
    TrainMetrics,
    TrainResponse,
)

SINGLE_CLIENT_EXAMPLE = {
    "monthly_fee": 9.99,
    "usage_hours": 27.92,
    "support_requests": 1,
    "account_age_months": 14,
    "failed_payments": 1,
    "region": "america",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1,
}

RISKY_CLIENT_EXAMPLE = {
    "monthly_fee": 49.99,
    "usage_hours": 1.5,
    "support_requests": 4,
    "account_age_months": 1,
    "failed_payments": 2,
    "region": "africa",
    "device_type": "mobile",
    "payment_method": "crypto",
    "autopay_enabled": 0,
}

PREDICT_EXAMPLES = {
    "single": {
        "summary": "Один клиент",
        "description": "Объект передаётся как есть, ответ всё равно содержит список из одного прогноза.",
        "value": SINGLE_CLIENT_EXAMPLE,
    },
    "batch": {
        "summary": "Список клиентов",
        "description": "Лояльный клиент и клиент из группы риска в одном запросе.",
        "value": [SINGLE_CLIENT_EXAMPLE, RISKY_CLIENT_EXAMPLE],
    },
}

TRAIN_EXAMPLES = {
    "logreg": {
        "summary": "Логистическая регрессия",
        "description": "Поведение по умолчанию, тело запроса можно не передавать вовсе.",
        "value": {"model_type": "logreg", "hyperparameters": {}},
    },
    "random_forest": {
        "summary": "Случайный лес",
        "description": "Другой тип модели со своими гиперпараметрами.",
        "value": {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 200, "max_depth": 8},
        },
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = load_churn_model()
    if artifact is not None:
        set_artifact(artifact)
    yield


app = FastAPI(title="ML Churn Service", lifespan=lifespan)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # отклонённое значение попадает в ответ, а nan и inf не сериализуются в json
    return JSONResponse(
        status_code=422,
        content={"detail": json_safe(jsonable_encoder(exc.errors()))},
    )


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
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"object at position {position}: {field} "
                        "must be a finite number"
                    ),
                )


@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@app.post("/predict", response_model=PredictionResponseChurn)
def predict(
    payload: FeatureVectorChurn | list[FeatureVectorChurn] = Body(
        ..., openapi_examples=PREDICT_EXAMPLES
    ),
):
    pipeline = get_model()
    if pipeline is None:
        raise HTTPException(
            status_code=409,
            detail="model is not trained yet, call POST /model/train first",
        )

    objects = payload if isinstance(payload, list) else [payload]
    if not objects:
        raise HTTPException(status_code=400, detail="no objects to predict")

    ensure_finite(objects)

    X = pd.DataFrame([obj.model_dump() for obj in objects])
    classes = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    stay_index = list(pipeline.classes_).index(0)
    churn_index = list(pipeline.classes_).index(1)

    predictions = [
        PredictionChurn(
            churn=int(cls),
            probability_stay=round(float(row[stay_index]), 4),
            probability_churn=round(float(row[churn_index]), 4),
        )
        for cls, row in zip(classes, proba)
    ]
    return PredictionResponseChurn(n_objects=len(predictions), predictions=predictions)


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
def model_train(
    df: Dataset,
    config: TrainingConfigChurn | None = Body(
        default=None, openapi_examples=TRAIN_EXAMPLES
    ),
):
    if df.empty:
        raise HTTPException(status_code=400, detail="dataset is empty")

    config = config or TrainingConfigChurn()
    try:
        pipeline, metrics, applied = train_churn_model(
            df, config.model_type, config.hyperparameters
        )
    except InvalidTrainingConfig as error:
        raise HTTPException(status_code=422, detail=str(error))

    artifact = ModelArtifact(
        pipeline=pipeline,
        trained_at=datetime.now(timezone.utc),
        metrics=metrics,
        model_type=config.model_type,
        hyperparameters=applied,
    )
    save_churn_model(artifact)
    set_artifact(artifact)
    return TrainResponse(
        status="trained",
        algorithm=type(pipeline.named_steps["classifier"]).__name__,
        model_type=config.model_type,
        hyperparameters=applied,
        n_train=metrics["n_train"],
        n_test=metrics["n_test"],
        metrics=TrainMetrics(accuracy=metrics["accuracy"], f1=metrics["f1"]),
    )


@app.get("/model/status", response_model=ModelStatus)
def model_status():
    artifact = get_artifact()
    if artifact is None:
        return ModelStatus(
            is_trained=False,
            trained_at=None,
            model_type=None,
            hyperparameters=None,
            metrics=None,
        )
    return ModelStatus(
        is_trained=True,
        trained_at=artifact.trained_at,
        # артефакты, сохранённые до дня 8, не содержат этих полей
        model_type=getattr(artifact, "model_type", None),
        hyperparameters=getattr(artifact, "hyperparameters", None),
        metrics=TrainMetrics(
            accuracy=artifact.metrics["accuracy"],
            f1=artifact.metrics["f1"],
        ),
    )
