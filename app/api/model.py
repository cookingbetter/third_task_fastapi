import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Query

from app.api.deps import Dataset, valid_records
from app.api.docs import METRICS_ERROR_RESPONSE, TRAIN_ERROR_RESPONSES, TRAIN_EXAMPLES
from app.core.config import HISTORY_LIMIT
from app.core.errors import (
    DATA_PREPARATION_FAILED,
    EMPTY_DATASET,
    INVALID_TARGET,
    VALIDATION_ERROR,
    InvalidTrainingConfig,
    ServiceError,
)
from app.ml.history import append_record, load_history
from app.ml.model import (
    MODEL_TYPES,
    ModelArtifact,
    get_artifact,
    save_churn_model,
    set_artifact,
    train_churn_model,
    trained_categories,
)
from app.ml.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    TARGET,
)
from app.schemas import (
    FeatureSchema,
    FeatureVectorChurn,
    MetricsHistoryResponse,
    ModelSchemaChurn,
    ModelStatus,
    TrainingConfigChurn,
    TrainMetrics,
    TrainResponse,
)

logger = logging.getLogger(__name__)

JSON_TYPES = {float: "number", int: "integer", str: "string"}

router = APIRouter(prefix="/model")


@router.post("/train", response_model=TrainResponse, responses=TRAIN_ERROR_RESPONSES)
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
    logger.info(
        "training started: model_type=%s hyperparameters=%s rows=%s",
        config.model_type,
        config.hyperparameters,
        len(df),
    )
    started = time.perf_counter()
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
    logger.info(
        "training finished in %.2fs: model_type=%s n_train=%s n_test=%s "
        "accuracy=%s f1=%s roc_auc=%s",
        time.perf_counter() - started,
        config.model_type,
        metrics["n_train"],
        metrics["n_test"],
        metrics["accuracy"],
        metrics["f1"],
        metrics["roc_auc"],
    )

    try:
        append_record(
            {
                "trained_at": artifact.trained_at.isoformat(),
                "model_type": config.model_type,
                "hyperparameters": applied,
                "metrics": {key: metrics[key] for key in ("accuracy", "f1", "roc_auc")},
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


@router.get(
    "/metrics",
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


@router.get("/schema", response_model=ModelSchemaChurn)
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


@router.get("/status", response_model=ModelStatus)
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
