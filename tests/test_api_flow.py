import pytest
from conftest import LOYAL_CLIENT, RISKY_CLIENT

from app.ml.dataset import dataset_state, get_dataset
from app.ml.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    TARGET,
    TEST_SIZE,
)


@pytest.fixture
def subset_client(client, with_dataset, real_subset):
    with_dataset(real_subset)
    return client


def test_root_reports_service_is_running(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "ml churn service is running"}


def test_status_reports_untrained_before_training(client):
    response = client.get("/model/status")

    assert response.status_code == 200
    assert response.json() == {
        "is_trained": False,
        "trained_at": None,
        "model_type": None,
        "hyperparameters": None,
        "metrics": None,
    }


def test_health_reports_untrained_service(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_available"] is False
    assert body["dataset_available"] is True


def test_health_reports_trained_model(subset_client):
    subset_client.post("/model/train")

    assert subset_client.get("/health").json()["model_available"] is True


def test_health_tracks_dataset_loading(client):
    # кеш датасета глобален, поэтому состояние задаём явно, а не надеемся на порядок тестов
    get_dataset.cache_clear()
    assert client.get("/health").json()["dataset_loaded"] is False

    client.get("/dataset/preview?n=1")

    body = client.get("/health").json()
    assert body["dataset_loaded"] is True
    assert body["n_rows"] == 2000


def test_dataset_state_without_file(tmp_path):
    state = dataset_state(tmp_path / "missing.csv")

    assert state["available"] is False


def test_dataset_info_describes_subset(subset_client, real_subset):
    response = subset_client.get("/dataset/info")

    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] == len(real_subset)
    assert body["n_columns"] == len(FEATURE_ORDER) + 1
    assert body["churn_distribution"] == {"0": 25, "1": 25}
    assert body["churn_share"] == {"0": 0.5, "1": 0.5}


def test_dataset_preview_returns_requested_rows(subset_client):
    response = subset_client.get("/dataset/preview?n=3")

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 3
    assert sorted(rows[0]) == sorted(FEATURE_ORDER + [TARGET])


def test_dataset_split_info_reports_stratified_split(subset_client, real_subset):
    response = subset_client.get("/dataset/split-info")

    assert response.status_code == 200
    body = response.json()
    assert body["test_size"] == TEST_SIZE
    assert body["stratified"] is True
    assert body["numeric_features"] == NUMERIC_FEATURES
    assert body["categorical_features"] == CATEGORICAL_FEATURES
    assert body["train"]["n_rows"] + body["test"]["n_rows"] == len(real_subset)
    assert body["test"]["churn_distribution"] == {"0": 5, "1": 5}


def test_train_status_predict_scenario(subset_client, real_subset):
    train = subset_client.post("/model/train")
    assert train.status_code == 200
    trained = train.json()
    assert trained["status"] == "trained"
    assert trained["algorithm"] == "LogisticRegression"
    assert trained["model_type"] == "logreg"
    assert trained["hyperparameters"] == {"max_iter": 1000, "random_state": 42}
    assert trained["n_train"] + trained["n_test"] == len(real_subset)

    status = subset_client.get("/model/status")
    assert status.status_code == 200
    assert status.json()["is_trained"] is True
    assert status.json()["trained_at"] is not None
    assert status.json()["model_type"] == "logreg"
    assert status.json()["metrics"] == trained["metrics"]

    single = subset_client.post("/predict", json=LOYAL_CLIENT)
    assert single.status_code == 200
    predicted = single.json()
    assert predicted["n_objects"] == 1
    prediction = predicted["predictions"][0]
    assert prediction["churn"] in (0, 1)
    assert prediction["probability_stay"] + prediction["probability_churn"] == pytest.approx(
        1.0, abs=2e-4
    )

    batch = subset_client.post("/predict", json=[LOYAL_CLIENT, RISKY_CLIENT])
    assert batch.status_code == 200
    assert batch.json()["n_objects"] == 2

    metrics = subset_client.get("/model/metrics")
    assert metrics.status_code == 200
    history = metrics.json()
    assert history["is_trained"] is True
    assert history["n_records"] == 1
    assert history["latest"]["model_type"] == "logreg"
    assert history["latest"]["metrics"] == trained["metrics"]
    assert history["latest"]["n_train"] == trained["n_train"]

    schema = subset_client.get("/model/schema")
    assert schema.status_code == 200
    assert schema.json()["is_trained"] is True
    assert schema.json()["feature_order"] == FEATURE_ORDER
    assert schema.json()["target"] == TARGET


def test_prediction_label_matches_the_larger_probability(client, with_dataset, synthetic_df):
    # в synthetic_df классы разделимы, поэтому ожидаем не только форму ответа,
    # но и совпадение метки с профилем клиента
    with_dataset(synthetic_df)
    client.post("/model/train")

    predictions = client.post("/predict", json=[LOYAL_CLIENT, RISKY_CLIENT]).json()["predictions"]

    loyal, risky = predictions
    assert loyal["churn"] == 0
    assert loyal["probability_stay"] > loyal["probability_churn"]
    assert risky["churn"] == 1
    assert risky["probability_churn"] > risky["probability_stay"]


def test_metrics_limit_restricts_returned_records(subset_client):
    for config in (
        {"model_type": "logreg", "hyperparameters": {"C": 0.5}},
        {"model_type": "logreg", "hyperparameters": {"C": 0.6}},
        {"model_type": "random_forest", "hyperparameters": {}},
    ):
        subset_client.post("/model/train", json=config)

    body = subset_client.get("/model/metrics?limit=2").json()

    assert body["n_records"] == 3
    assert len(body["records"]) == 2
    # срез берёт свежие записи, а порядок в ответе от новых к старым
    assert body["records"][0]["model_type"] == "random_forest"
    assert body["records"][1]["hyperparameters"]["C"] == 0.6


def test_model_schema_describes_trained_categories(subset_client):
    subset_client.post("/model/train")

    schema = subset_client.get("/model/schema").json()
    features = {item["name"]: item for item in schema["features"]}

    for name in CATEGORICAL_FEATURES:
        assert features[name]["kind"] == "categorical"
        assert features[name]["allowed_values"]
    for name in NUMERIC_FEATURES:
        assert features[name]["kind"] == "numeric"
        assert features[name]["allowed_values"] is None


def test_training_random_forest_is_recorded_separately(subset_client):
    subset_client.post("/model/train", json={"model_type": "logreg", "hyperparameters": {}})
    forest = subset_client.post(
        "/model/train",
        json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 20}},
    )

    assert forest.status_code == 200
    assert forest.json()["algorithm"] == "RandomForestClassifier"
    assert forest.json()["hyperparameters"] == {"random_state": 42, "n_estimators": 20}

    history = subset_client.get("/model/metrics").json()
    assert history["n_records"] == 2
    assert history["latest"]["model_type"] == "random_forest"
    assert [item["model_type"] for item in history["records"]] == ["random_forest", "logreg"]

    filtered = subset_client.get("/model/metrics?model_type=logreg").json()
    assert filtered["n_records"] == 1
    assert filtered["latest"]["model_type"] == "logreg"


def test_training_the_same_data_twice_gives_the_same_metrics(subset_client):
    first = subset_client.post("/model/train").json()
    second = subset_client.post("/model/train").json()

    assert first["metrics"] == second["metrics"]
    assert first["n_train"] == second["n_train"]


def test_prediction_survives_model_reload(subset_client, isolated_storage):
    from app.ml.model import load_churn_model, set_artifact

    subset_client.post("/model/train")
    before = subset_client.post("/predict", json=RISKY_CLIENT).json()

    # имитируем перезапуск сервиса: артефакт берётся с диска, как это делает lifespan
    set_artifact(load_churn_model(isolated_storage["model"]))
    after = subset_client.post("/predict", json=RISKY_CLIENT).json()

    assert after == before


def test_real_dataset_file_is_readable(client):
    response = client.get("/dataset/info")

    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] == 2000
    assert body["n_columns"] == len(FEATURE_ORDER) + 1
    assert sorted(body["churn_distribution"]) == ["0", "1"]
