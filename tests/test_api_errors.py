import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from conftest import LOYAL_CLIENT
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.api import deps
from app.core.errors import (
    DATA_PREPARATION_FAILED,
    DATASET_NOT_FOUND,
    EMPTY_DATASET,
    EMPTY_PAYLOAD,
    FEATURE_MISMATCH,
    INCOMPATIBLE_MODEL,
    INVALID_TARGET,
    INVALID_TRAINING_CONFIG,
    MODEL_NOT_TRAINED,
    NON_FINITE_VALUE,
    VALIDATION_ERROR,
)
from app.ml.model import ModelArtifact, build_pipeline, set_artifact
from app.ml.preprocessing import FEATURE_ORDER, NUMERIC_FEATURES, TARGET, prepare_data

JSON_HEADERS = {"content-type": "application/json"}


def assert_error(response, status_code: int, code: str) -> dict:
    assert response.status_code == status_code
    body = response.json()
    assert sorted(body) == ["code", "details", "message"]
    assert body["code"] == code
    assert isinstance(body["message"], str) and body["message"]
    assert isinstance(body["details"], dict)
    return body


def install_artifact(pipeline) -> None:
    set_artifact(
        ModelArtifact(
            pipeline=pipeline,
            trained_at=datetime.now(timezone.utc),
            metrics={"accuracy": 1.0, "f1": 1.0, "roc_auc": 1.0},
        )
    )


@pytest.fixture
def subset_client(client, with_dataset, real_subset):
    with_dataset(real_subset)
    return client


@pytest.fixture
def trained_client(subset_client):
    assert subset_client.post("/model/train").status_code == 200
    return subset_client


def test_predict_without_trained_model(client):
    assert_error(client.post("/predict", json=LOYAL_CLIENT), 409, MODEL_NOT_TRAINED)


def test_predict_with_empty_list(trained_client):
    assert_error(trained_client.post("/predict", json=[]), 400, EMPTY_PAYLOAD)


def test_predict_with_extra_field(trained_client):
    payload = {**LOYAL_CLIENT, "loyalty_points": 10}

    body = assert_error(trained_client.post("/predict", json=payload), 422, VALIDATION_ERROR)

    assert body["details"]["errors"][0]["type"] == "extra_forbidden"


def test_predict_with_missing_field(trained_client):
    payload = {key: value for key, value in LOYAL_CLIENT.items() if key != "monthly_fee"}

    body = assert_error(trained_client.post("/predict", json=payload), 422, VALIDATION_ERROR)

    assert body["details"]["errors"][0]["type"] == "missing"


def test_predict_with_wrong_type(trained_client):
    payload = {**LOYAL_CLIENT, "support_requests": "many"}

    assert_error(trained_client.post("/predict", json=payload), 422, VALIDATION_ERROR)


@pytest.mark.parametrize("batch_size", [10, 500])
def test_validation_error_does_not_echo_the_payload(trained_client, batch_size):
    # один битый объект в батче: ответ не должен расти вместе с размером запроса
    payload = [LOYAL_CLIENT] * (batch_size - 1) + [{**LOYAL_CLIENT, "loyalty_points": 1}]

    body = assert_error(trained_client.post("/predict", json=payload), 422, VALIDATION_ERROR)

    errors = body["details"]["errors"]
    # payload объявлен как объект или список, поэтому pydantic отчитывается по обеим ветвям
    from_list = [error for error in errors if "list[FeatureVectorChurn]" in error["loc"]]
    assert len(from_list) == 1
    assert from_list[0]["type"] == "extra_forbidden"
    assert all("input" not in error for error in errors)
    assert len(json.dumps(body)) < 700


def test_predict_with_non_finite_feature(trained_client):
    payload = json.dumps({**LOYAL_CLIENT, "monthly_fee": float("inf")})

    body = assert_error(
        trained_client.post("/predict", content=payload, headers=JSON_HEADERS),
        422,
        NON_FINITE_VALUE,
    )

    assert body["details"] == {"position": 0, "field": "monthly_fee"}


def test_predict_with_malformed_json(trained_client):
    assert_error(
        trained_client.post("/predict", content="{not json", headers=JSON_HEADERS),
        422,
        VALIDATION_ERROR,
    )


def test_predict_with_model_trained_on_other_features(client, real_subset):
    columns = ["monthly_fee", "usage_hours", "support_requests"]
    pipeline = Pipeline(
        [
            ("preprocessor", ColumnTransformer([("numeric", StandardScaler(), columns)])),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(real_subset[columns], real_subset[TARGET])
    install_artifact(pipeline)

    body = assert_error(client.post("/predict", json=LOYAL_CLIENT), 409, FEATURE_MISMATCH)

    assert body["details"] == {"model_expects": columns, "service_provides": FEATURE_ORDER}


def test_predict_with_model_trained_without_feature_names(client, real_subset):
    # модель, обученную на массиве без имён столбцов, сверить с FEATURE_ORDER нельзя,
    # поэтому сервис обязан признать её несовместимой, а не доверять ей молча
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(real_subset[NUMERIC_FEATURES].to_numpy(), real_subset[TARGET])
    assert not hasattr(pipeline, "feature_names_in_")
    install_artifact(pipeline)

    body = assert_error(client.post("/predict", json=LOYAL_CLIENT), 409, FEATURE_MISMATCH)

    assert body["details"] == {"service_provides": FEATURE_ORDER}


def test_predict_with_model_trained_on_other_labels(client, real_subset):
    shifted = real_subset.copy()
    shifted[TARGET] = shifted[TARGET] + 1
    X, y = prepare_data(shifted)
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    install_artifact(pipeline)

    body = assert_error(client.post("/predict", json=LOYAL_CLIENT), 409, INCOMPATIBLE_MODEL)

    assert body["details"] == {"model_classes": [1, 2], "service_expects": [0, 1]}


def test_train_with_empty_dataset(client, with_dataset):
    with_dataset(pd.DataFrame())

    assert_error(client.post("/model/train"), 400, EMPTY_DATASET)


def test_train_without_dataset_file(client, monkeypatch):
    def missing_file():
        raise FileNotFoundError("data/churn_dataset.csv")

    monkeypatch.setattr(deps, "get_dataset", missing_file)

    assert_error(client.post("/model/train"), 404, DATASET_NOT_FOUND)


def test_train_with_unreadable_dataset(client, monkeypatch):
    def broken_file():
        raise ValueError("1 validation error for DatasetRowChurn")

    monkeypatch.setattr(deps, "get_dataset", broken_file)

    assert_error(client.post("/model/train"), 422, DATA_PREPARATION_FAILED)


def test_train_with_unexpected_target_labels(client, with_dataset, real_subset):
    shifted = real_subset.copy()
    shifted[TARGET] = shifted[TARGET] + 1
    with_dataset(shifted)

    body = assert_error(client.post("/model/train"), 422, INVALID_TARGET)

    assert body["details"] == {"found_labels": [1, 2]}


def test_train_with_single_member_class(client, with_dataset, real_subset):
    almost_one_class = pd.concat(
        [real_subset[real_subset[TARGET] == 0], real_subset[real_subset[TARGET] == 1].head(1)],
        ignore_index=True,
    )
    with_dataset(almost_one_class)

    assert_error(client.post("/model/train"), 422, DATA_PREPARATION_FAILED)


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "xgboost", "hyperparameters": {}},
        {"model_type": "logreg", "hyperparameters": {"depth": 3}},
        {"model_type": "logreg", "hyperparameters": {"C": -1}},
    ],
)
def test_train_with_invalid_config(subset_client, config):
    assert_error(subset_client.post("/model/train", json=config), 422, INVALID_TRAINING_CONFIG)


def test_train_with_non_finite_hyperparameter(subset_client):
    payload = json.dumps({"model_type": "logreg", "hyperparameters": {"C": float("inf")}})

    assert_error(
        subset_client.post("/model/train", content=payload, headers=JSON_HEADERS),
        422,
        INVALID_TRAINING_CONFIG,
    )


def test_train_ignores_unknown_config_field(subset_client):
    # TrainingConfigChurn не запрещает лишние поля, в отличие от вектора признаков
    payload = {"model_type": "logreg", "hyperparameters": {}, "epochs": 5}

    response = subset_client.post("/model/train", json=payload)

    assert response.status_code == 200
    assert response.json()["model_type"] == "logreg"


def test_metrics_with_unknown_model_type(client):
    body = assert_error(client.get("/model/metrics?model_type=xgboost"), 422, VALIDATION_ERROR)

    assert body["details"] == {"available": ["logreg", "random_forest"]}


@pytest.mark.parametrize("limit", [0, -1, 101, "many"])
def test_metrics_with_limit_out_of_range(client, limit):
    assert_error(client.get(f"/model/metrics?limit={limit}"), 422, VALIDATION_ERROR)


def test_metrics_without_history(client):
    response = client.get("/model/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "is_trained": False,
        "latest": None,
        "n_records": 0,
        "records": [],
    }


def test_unknown_path_returns_not_found(client):
    assert_error(client.get("/model/nope"), 404, "not_found")


def test_wrong_method_returns_method_not_allowed(client):
    assert_error(client.delete("/model/metrics"), 405, "method_not_allowed")


def test_errors_never_leak_tracebacks(trained_client):
    responses = [
        trained_client.post("/predict", json=[]),
        trained_client.post("/predict", json={"monthly_fee": 1.0}),
        trained_client.get("/model/metrics?limit=0"),
        trained_client.get("/model/nope"),
    ]

    for response in responses:
        text = response.text
        assert "Traceback" not in text
        assert "app/" not in text
        assert ".py" not in text
