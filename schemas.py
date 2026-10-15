from datetime import datetime

from pydantic import BaseModel


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class DatasetInfo(BaseModel):
    n_rows: int
    n_columns: int
    features: list[str]
    churn_distribution: dict[int, int]
    churn_share: dict[int, float]


class ClassBalance(BaseModel):
    n_rows: int
    churn_distribution: dict[int, int]
    churn_share: dict[int, float]


class SplitInfo(BaseModel):
    test_size: float
    random_state: int
    stratified: bool
    numeric_features: list[str]
    categorical_features: list[str]
    train: ClassBalance
    test: ClassBalance


class TrainMetrics(BaseModel):
    accuracy: float
    f1: float


class TrainResponse(BaseModel):
    status: str
    algorithm: str
    n_train: int
    n_test: int
    metrics: TrainMetrics


class ModelStatus(BaseModel):
    is_trained: bool
    trained_at: datetime | None
    metrics: TrainMetrics | None
