import pandas as pd
import pytest
from fastapi.testclient import TestClient

import history
import main
import model
from dataset import load_dataset
from preprocessing import TARGET

LOYAL_CLIENT = {
    "monthly_fee": 9.99,
    "usage_hours": 27.92,
    "support_requests": 1,
    "account_age_months": 14,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1,
}

RISKY_CLIENT = {
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


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    # пути в save_churn_model и append_record зафиксированы значениями по умолчанию
    # ещё при определении функций, поэтому подменяем сами функции в main
    paths = {
        "model": tmp_path / "churn_model.joblib",
        "history": tmp_path / "training_history.json",
    }
    monkeypatch.setattr(model, "_artifact", None)
    monkeypatch.setattr(
        main,
        "save_churn_model",
        lambda artifact: model.save_churn_model(artifact, paths["model"]),
    )
    monkeypatch.setattr(
        main,
        "append_record",
        lambda record: history.append_record(record, paths["history"]),
    )
    monkeypatch.setattr(
        main, "load_history", lambda: history.load_history(paths["history"])
    )
    return paths


@pytest.fixture
def client():
    # без контекстного менеджера lifespan не запускается,
    # поэтому модель с диска не подхватывается и тесты видят необученный сервис
    return TestClient(main.app)


@pytest.fixture
def with_dataset():
    def override(df: pd.DataFrame) -> pd.DataFrame:
        main.app.dependency_overrides[main.dataset_dependency] = lambda: df
        return df

    yield override
    main.app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def real_dataset():
    return load_dataset()


@pytest.fixture
def real_subset(real_dataset):
    # по 25 первых строк каждого класса: подвыборка не зависит от сида
    return pd.concat(
        [
            real_dataset[real_dataset[TARGET] == 0].head(25),
            real_dataset[real_dataset[TARGET] == 1].head(25),
        ],
        ignore_index=True,
    )


def synthetic_rows(n_per_class: int = 30) -> list[dict]:
    rows = []
    for index in range(n_per_class):
        # churn явно зависит от признаков, иначе метрики плавали бы от запуска к запуску
        rows.append(
            {
                "monthly_fee": 10.0 + index % 5,
                "usage_hours": 40.0 + index % 7,
                "support_requests": 0,
                "account_age_months": 24 + index % 6,
                "failed_payments": 0,
                "region": "europe",
                "device_type": "desktop",
                "payment_method": "card",
                "autopay_enabled": 1,
                "churn": 0,
            }
        )
        rows.append(
            {
                "monthly_fee": 70.0 + index % 5,
                "usage_hours": 2.0 + index % 3,
                "support_requests": 5 + index % 3,
                "account_age_months": 1 + index % 2,
                "failed_payments": 3,
                "region": "africa",
                "device_type": "mobile",
                "payment_method": "crypto",
                "autopay_enabled": 0,
                "churn": 1,
            }
        )
    return rows


@pytest.fixture
def synthetic_df():
    return pd.DataFrame(synthetic_rows())
