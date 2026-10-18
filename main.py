import logging
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from dataset import get_dataset
from errors import (
    DATA_PREPARATION_FAILED,
    DATASET_NOT_FOUND,
    EMPTY_DATASET,
    EMPTY_PAYLOAD,
    FEATURE_MISMATCH,
    INCOMPATIBLE_MODEL,
    INTERNAL_ERROR,
    INVALID_TARGET,
    INVALID_TRAINING_CONFIG,
    MODEL_NOT_TRAINED,
    NON_FINITE_VALUE,
    PREDICTION_FAILED,
    STATUS_CODES,
    VALIDATION_ERROR,
    ServiceError,
    error_payload,
)
from history import HISTORY_LIMIT, append_record, load_history
from model import (
    MODEL_TYPES,
    InvalidTrainingConfig,
    ModelArtifact,
    get_artifact,
    get_model,
    load_churn_model,
    save_churn_model,
    set_artifact,
    train_churn_model,
    trained_categories,
)
from preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    class_distribution,
    features_to_frame,
    prepare_data,
    split_train_test,
)
from schemas import (
    ClassBalance,
    DatasetInfo,
    DatasetRowChurn,
    ErrorResponse,
    FeatureSchema,
    FeatureVectorChurn,
    MetricsHistoryResponse,
    ModelSchemaChurn,
    ModelStatus,
    PredictionChurn,
    PredictionResponseChurn,
    SplitInfo,
    TrainingConfigChurn,
    TrainingRecord,
    TrainMetrics,
    TrainResponse,
)

logger = logging.getLogger(__name__)

JSON_TYPES = {float: "number", int: "integer", str: "string"}

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


def error_example(
    summary: str, code: str, message: str, details: dict | None = None
) -> dict:
    return {
        "summary": summary,
        "value": {"code": code, "message": message, "details": details or {}},
    }


def error_responses(description: str, examples: dict) -> dict:
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {"application/json": {"examples": examples}},
    }


DATASET_NOT_FOUND_RESPONSE = error_responses(
    "Файл датасета не найден",
    {
        "dataset_not_found": error_example(
            "Нет файла датасета", DATASET_NOT_FOUND, "dataset file not found"
        )
    },
)

EMPTY_DATASET_RESPONSE = error_responses(
    "Датасет пуст",
    {"empty_dataset": error_example("Пустой датасет", EMPTY_DATASET, "dataset is empty")},
)

DATASET_LOAD_EXAMPLE = error_example(
    "Файл датасета не читается",
    DATA_PREPARATION_FAILED,
    "could not load dataset: 1 validation error for DatasetRowChurn",
)

SPLIT_ERROR_RESPONSE = error_responses(
    "Датасет не удалось подготовить или разбить",
    {
        "dataset_load_failed": DATASET_LOAD_EXAMPLE,
        "data_preparation_failed": error_example(
            "Класс слишком мал для стратификации",
            DATA_PREPARATION_FAILED,
            "could not prepare dataset: The least populated class in y has only 1 member",
        ),
    },
)

PREVIEW_ERROR_RESPONSE = error_responses(
    "Неверный параметр запроса или нечитаемый датасет",
    {
        "validation_error": error_example(
            "Параметр n вне допустимого диапазона",
            VALIDATION_ERROR,
            "request does not match the expected schema",
            {
                "errors": [
                    {
                        "type": "less_than_equal",
                        "loc": ["query", "n"],
                        "msg": "Input should be less than or equal to 100",
                    }
                ]
            },
        ),
        "dataset_load_failed": DATASET_LOAD_EXAMPLE,
    },
)

METRICS_ERROR_RESPONSE = error_responses(
    "Неверный фильтр или параметр запроса",
    {
        "unknown_model_type": error_example(
            "Неизвестный тип модели в фильтре",
            VALIDATION_ERROR,
            "unknown model_type 'xgboost'",
            {"available": ["logreg", "random_forest"]},
        ),
        "validation_error": error_example(
            "Параметр limit вне допустимого диапазона",
            VALIDATION_ERROR,
            "request does not match the expected schema",
            {
                "errors": [
                    {
                        "type": "greater_than_equal",
                        "loc": ["query", "limit"],
                        "msg": "Input should be greater than or equal to 1",
                    }
                ]
            },
        ),
    },
)

PREDICT_ERROR_RESPONSES = {
    400: error_responses(
        "Передан пустой список объектов",
        {
            "empty_payload": error_example(
                "Пустой список", EMPTY_PAYLOAD, "no objects to predict"
            )
        },
    ),
    409: error_responses(
        "Модель не обучена или обучена на другом наборе признаков",
        {
            "model_not_trained": error_example(
                "Модель не обучена",
                MODEL_NOT_TRAINED,
                "model is not trained yet, call POST /model/train first",
            ),
            "feature_mismatch": error_example(
                "Другой набор признаков",
                FEATURE_MISMATCH,
                "model was trained on different features, retrain it",
                {
                    "model_expects": ["monthly_fee", "usage_hours", "region"],
                    "service_provides": FEATURE_ORDER,
                },
            ),
            "incompatible_model": error_example(
                "Модель предсказывает другие классы",
                INCOMPATIBLE_MODEL,
                "model predicts unexpected classes, retrain it",
                {"model_classes": [1, 2], "service_expects": [0, 1]},
            ),
        },
    ),
    422: error_responses(
        "Неверная структура запроса, типы значений или нечисловые признаки",
        {
            "validation_error": error_example(
                "Лишний или пропущенный признак",
                VALIDATION_ERROR,
                "request does not match the expected schema",
                {
                    "errors": [
                        {
                            "type": "extra_forbidden",
                            "loc": ["body", "extra_feature"],
                            "msg": "Extra inputs are not permitted",
                        }
                    ]
                },
            ),
            "non_finite_value": error_example(
                "NaN или бесконечность в признаке",
                NON_FINITE_VALUE,
                "monthly_fee must be a finite number",
                {"position": 0, "field": "monthly_fee"},
            ),
            "prediction_failed": error_example(
                "Модель не смогла обработать признаки",
                PREDICTION_FAILED,
                "model could not process the request: X has 9 features, "
                "but ColumnTransformer is expecting 3 features as input.",
            ),
        },
    ),
}

TRAIN_ERROR_RESPONSES = {
    400: error_responses(
        "Датасет пуст",
        {"empty_dataset": error_example("Пустой датасет", EMPTY_DATASET, "dataset is empty")},
    ),
    404: DATASET_NOT_FOUND_RESPONSE,
    422: error_responses(
        "Неверная конфигурация обучения или непригодные данные",
        {
            "invalid_training_config": error_example(
                "Неизвестный тип модели",
                INVALID_TRAINING_CONFIG,
                "unknown model_type 'xgboost', available: logreg, random_forest",
            ),
            "unknown_hyperparameters": error_example(
                "Неизвестный гиперпараметр",
                INVALID_TRAINING_CONFIG,
                "unknown hyperparameters for 'logreg': depth; allowed: C, class_weight, dual",
            ),
            "invalid_target": error_example(
                "Метки целевой переменной не 0 и 1",
                INVALID_TARGET,
                f"target '{TARGET}' must contain exactly labels 0 and 1",
                {"found_labels": [1, 2]},
            ),
            "data_preparation_failed": error_example(
                "Сбой подготовки данных",
                DATA_PREPARATION_FAILED,
                "could not prepare dataset for training: "
                "The least populated class in y has only 1 member",
            ),
        },
    ),
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


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(exc.code, exc.message, exc.details),
    )


@app.exception_handler(InvalidTrainingConfig)
async def training_config_error_handler(request: Request, exc: InvalidTrainingConfig):
    return JSONResponse(
        status_code=422,
        content=error_payload(INVALID_TRAINING_CONFIG, str(exc)),
    )


def compact_errors(errors) -> list[dict]:
    # присланное значение не возвращаем: на большом батче оно раздувает ответ
    return [
        {
            "type": error.get("type"),
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg"),
        }
        for error in errors
    ]


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=error_payload(
            VALIDATION_ERROR,
            "request does not match the expected schema",
            {"errors": json_safe(jsonable_encoder(compact_errors(exc.errors())))},
        ),
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    # сюда попадает то, что поднимает сам fastapi: неизвестный путь, чужой метод
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            STATUS_CODES.get(exc.status_code, "http_error"), str(exc.detail)
        ),
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled error while processing %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content=error_payload(INTERNAL_ERROR, "internal server error"),
    )


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


@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@app.post(
    "/predict",
    response_model=PredictionResponseChurn,
    responses=PREDICT_ERROR_RESPONSES,
)
def predict(
    payload: FeatureVectorChurn | list[FeatureVectorChurn] = Body(
        ..., openapi_examples=PREDICT_EXAMPLES
    ),
):
    pipeline = get_model()
    if pipeline is None:
        raise ServiceError(
            409,
            MODEL_NOT_TRAINED,
            "model is not trained yet, call POST /model/train first",
        )

    objects = payload if isinstance(payload, list) else [payload]
    if not objects:
        raise ServiceError(400, EMPTY_PAYLOAD, "no objects to predict")

    ensure_finite(objects)

    names = getattr(pipeline, "feature_names_in_", None)
    if names is None:
        # модель обучена без имён столбцов, сверить набор признаков невозможно
        raise ServiceError(
            409,
            FEATURE_MISMATCH,
            "model was trained without feature names, retrain it",
            {"service_provides": FEATURE_ORDER},
        )

    expected = list(names)
    if expected != FEATURE_ORDER:
        raise ServiceError(
            409,
            FEATURE_MISMATCH,
            "model was trained on different features, retrain it",
            {"model_expects": expected, "service_provides": FEATURE_ORDER},
        )

    # numpy-метки не сериализуются в json, поэтому приводим их к int сразу
    model_classes = [int(value) for value in pipeline.classes_]
    if sorted(model_classes) != [0, 1]:
        raise ServiceError(
            409,
            INCOMPATIBLE_MODEL,
            "model predicts unexpected classes, retrain it",
            {"model_classes": model_classes, "service_expects": [0, 1]},
        )

    X = features_to_frame([obj.model_dump() for obj in objects])
    try:
        classes = pipeline.predict(X)
        proba = pipeline.predict_proba(X)
    except (KeyError, ValueError) as error:
        raise ServiceError(
            422,
            PREDICTION_FAILED,
            f"model could not process the request: {error}",
        )

    stay_index = model_classes.index(0)
    churn_index = model_classes.index(1)

    predictions = [
        PredictionChurn(
            churn=int(cls),
            probability_stay=round(float(row[stay_index]), 4),
            probability_churn=round(float(row[churn_index]), 4),
        )
        for cls, row in zip(classes, proba)
    ]
    return PredictionResponseChurn(n_objects=len(predictions), predictions=predictions)


@app.get(
    "/dataset/preview",
    response_model=list[DatasetRowChurn],
    responses={
        404: DATASET_NOT_FOUND_RESPONSE,
        422: PREVIEW_ERROR_RESPONSE,
    },
)
def dataset_preview(df: Dataset, n: int = Query(5, ge=1, le=100)):
    return df.head(n).to_dict(orient="records")


@app.get(
    "/dataset/info",
    response_model=DatasetInfo,
    responses={
        400: EMPTY_DATASET_RESPONSE,
        404: DATASET_NOT_FOUND_RESPONSE,
        422: error_responses(
            "Файл датасета не читается", {"dataset_load_failed": DATASET_LOAD_EXAMPLE}
        ),
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


@app.get(
    "/dataset/split-info",
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


@app.post(
    "/model/train", response_model=TrainResponse, responses=TRAIN_ERROR_RESPONSES
)
def model_train(
    df: Dataset,
    config: TrainingConfigChurn | None = Body(
        default=None, openapi_examples=TRAIN_EXAMPLES
    ),
):
    if df.empty:
        raise ServiceError(400, EMPTY_DATASET, "dataset is empty")

    # на других метках модель обучится, но предсказывать по ней будет нечем
    labels = sorted(int(value) for value in df[TARGET].dropna().unique())
    if labels != [0, 1]:
        raise ServiceError(
            422,
            INVALID_TARGET,
            f"target '{TARGET}' must contain exactly labels 0 and 1",
            {"found_labels": labels},
        )

    config = config or TrainingConfigChurn()
    try:
        pipeline, metrics, applied = train_churn_model(
            df, config.model_type, config.hyperparameters
        )
    except InvalidTrainingConfig:
        # обрабатывается глобально: дело в конфигурации, а не в данных
        raise
    except (KeyError, ValueError) as error:
        raise ServiceError(
            422,
            DATA_PREPARATION_FAILED,
            f"could not prepare dataset for training: {error}",
        )

    artifact = ModelArtifact(
        pipeline=pipeline,
        trained_at=datetime.now(timezone.utc),
        metrics=metrics,
        model_type=config.model_type,
        hyperparameters=applied,
    )
    save_churn_model(artifact)
    set_artifact(artifact)

    try:
        append_record(
            {
                "trained_at": artifact.trained_at.isoformat(),
                "model_type": config.model_type,
                "hyperparameters": applied,
                "metrics": {
                    key: metrics[key] for key in ("accuracy", "f1", "roc_auc")
                },
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
            }
        )
    except (OSError, TypeError, ValueError):
        # обучение уже прошло, история вторична и не должна его отменять
        logger.warning("could not append training history", exc_info=True)

    return TrainResponse(
        status="trained",
        algorithm=type(pipeline.named_steps["classifier"]).__name__,
        model_type=config.model_type,
        hyperparameters=applied,
        n_train=metrics["n_train"],
        n_test=metrics["n_test"],
        metrics=TrainMetrics(
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            roc_auc=metrics["roc_auc"],
        ),
    )


@app.get(
    "/model/metrics",
    response_model=MetricsHistoryResponse,
    responses={422: METRICS_ERROR_RESPONSE},
)
def model_metrics(
    limit: int = Query(5, ge=1, le=HISTORY_LIMIT),
    model_type: str | None = Query(default=None),
):
    if model_type is not None and model_type not in MODEL_TYPES:
        raise ServiceError(
            422,
            VALIDATION_ERROR,
            f"unknown model_type '{model_type}'",
            {"available": sorted(MODEL_TYPES)},
        )

    records = [
        record
        for record in valid_records(load_history())
        if model_type is None or record.model_type == model_type
    ]
    return MetricsHistoryResponse(
        is_trained=get_artifact() is not None,
        latest=records[-1] if records else None,
        n_records=len(records),
        records=list(reversed(records[-limit:])),
    )


@app.get("/model/schema", response_model=ModelSchemaChurn)
def model_schema():
    artifact = get_artifact()
    categories = trained_categories(artifact.pipeline) if artifact else {}
    features = [
        FeatureSchema(
            name=name,
            type=JSON_TYPES[FeatureVectorChurn.model_fields[name].annotation],
            kind="numeric" if name in NUMERIC_FEATURES else "categorical",
            allowed_values=categories.get(name),
        )
        for name in FEATURE_ORDER
    ]
    return ModelSchemaChurn(
        target=TARGET,
        is_trained=artifact is not None,
        feature_order=FEATURE_ORDER,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        features=features,
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
            # артефакты, сохранённые до дня 11, не содержат roc_auc
            roc_auc=artifact.metrics.get("roc_auc"),
        ),
    )
