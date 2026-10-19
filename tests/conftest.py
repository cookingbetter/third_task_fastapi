import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import app
from app.ml import history, model
from app.ml.dataset import load_dataset
from app.ml.preprocessing import TARGET

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
    # подменяем сами пути, а не функции: тогда изоляция не зависит от того,
    # каким способом роутер вызывает сохранение модели и записи истории
    paths = {
        "model": tmp_path / "churn_model.joblib",
        "history": tmp_path / "training_history.json",
    }
    monkeypatch.setattr(model, "_artifact", None)
    monkeypatch.setattr(model, "MODEL_PATH", paths["model"])
    monkeypatch.setattr(history, "HISTORY_PATH", paths["history"])
    return paths


@pytest.fixture
def client():
    # без контекстного менеджера lifespan не запускается,
    # поэтому модель с диска не подхватывается и тесты видят необученный сервис
    return TestClient(app)


@pytest.fixture
def with_dataset():
    def override(df: pd.DataFrame) -> pd.DataFrame:
        app.dependency_overrides[deps.dataset_dependency] = lambda: df
        return df

    yield override
    app.dependency_overrides.clear()


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
