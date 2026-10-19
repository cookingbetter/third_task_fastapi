import logging

from fastapi import APIRouter, Body

from app.api.deps import ensure_finite
from app.api.docs import PREDICT_ERROR_RESPONSES, PREDICT_EXAMPLES
from app.core.errors import (
    EMPTY_PAYLOAD,
    FEATURE_MISMATCH,
    INCOMPATIBLE_MODEL,
    MODEL_NOT_TRAINED,
    PREDICTION_FAILED,
    ServiceError,
)
from app.ml.model import get_model
from app.ml.preprocessing import FEATURE_ORDER, features_to_frame
from app.schemas import FeatureVectorChurn, PredictionChurn, PredictionResponseChurn

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
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
    logger.info(
        "prediction served: n_objects=%s churn_predicted=%s",
        len(predictions),
        sum(item.churn for item in predictions),
    )
    return PredictionResponseChurn(n_objects=len(predictions), predictions=predictions)
