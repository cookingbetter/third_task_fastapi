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
        # лишний признак — ошибка: модель всё равно не умеет его учитывать
        "extra": "forbid",
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

    # запрет лишних полей нужен запросам, а в файле датасета лишний столбец допустим
    model_config = {"extra": "ignore"}


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


class TrainingConfigChurn(BaseModel):
    model_type: str = "logreg"
    hyperparameters: dict = {}

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_type": "random_forest",
                    "hyperparameters": {"n_estimators": 200, "max_depth": 8},
                }
            ]
        }
    }


class TrainMetrics(BaseModel):
    accuracy: float
    f1: float
    # артефакты и записи истории, созданные до дня 11, не содержат roc_auc
    roc_auc: float | None = None


class TrainResponse(BaseModel):
    status: str
    algorithm: str
    model_type: str
    hyperparameters: dict
    n_train: int
    n_test: int
    metrics: TrainMetrics


class ModelStatus(BaseModel):
    is_trained: bool
    trained_at: datetime | None
    model_type: str | None
    hyperparameters: dict | None
    metrics: TrainMetrics | None


class HealthStatus(BaseModel):
    status: str
    model_available: bool
    dataset_available: bool
    # датасет читается по первому запросу, поэтому наличие файла и загрузка в память различаются
    dataset_loaded: bool
    n_rows: int | None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "model_available": True,
                    "dataset_available": True,
                    "dataset_loaded": True,
                    "n_rows": 2000,
                }
            ]
        }
    }


class FeatureSchema(BaseModel):
    name: str
    type: str
    kind: str
    allowed_values: list[str] | None


class ModelSchemaChurn(BaseModel):
    target: str
    is_trained: bool
    feature_order: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    features: list[FeatureSchema]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "target": "churn",
                    "is_trained": True,
                    "feature_order": ["monthly_fee", "region"],
                    "numeric_features": ["monthly_fee"],
                    "categorical_features": ["region"],
                    "features": [
                        {
                            "name": "monthly_fee",
                            "type": "number",
                            "kind": "numeric",
                            "allowed_values": None,
                        },
                        {
                            "name": "region",
                            "type": "string",
                            "kind": "categorical",
                            "allowed_values": [
                                "africa",
                                "america",
                                "asia",
                                "europe",
                            ],
                        },
                    ],
                }
            ]
        }
    }


class TrainingRecord(BaseModel):
    trained_at: datetime
    model_type: str
    hyperparameters: dict
    metrics: TrainMetrics
    n_train: int
    n_test: int


class MetricsHistoryResponse(BaseModel):
    is_trained: bool
    latest: TrainingRecord | None
    n_records: int
    records: list[TrainingRecord]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "is_trained": True,
                    "latest": {
                        "trained_at": "2026-10-18T14:24:54.317330Z",
                        "model_type": "random_forest",
                        "hyperparameters": {
                            "random_state": 42,
                            "n_estimators": 200,
                            "max_depth": 8,
                        },
                        "metrics": {
                            "accuracy": 0.795,
                            "f1": 0.1458,
                            "roc_auc": 0.6421,
                        },
                        "n_train": 1600,
                        "n_test": 400,
                    },
                    "n_records": 2,
                    "records": [
                        {
                            "trained_at": "2026-10-18T14:24:54.317330Z",
                            "model_type": "random_forest",
                            "hyperparameters": {
                                "random_state": 42,
                                "n_estimators": 200,
                                "max_depth": 8,
                            },
                            "metrics": {
                                "accuracy": 0.795,
                                "f1": 0.1458,
                                "roc_auc": 0.6421,
                            },
                            "n_train": 1600,
                            "n_test": 400,
                        },
                        {
                            "trained_at": "2026-10-18T14:20:11.004512Z",
                            "model_type": "logreg",
                            "hyperparameters": {
                                "max_iter": 1000,
                                "random_state": 42,
                            },
                            "metrics": {
                                "accuracy": 0.7875,
                                "f1": 0.0449,
                                "roc_auc": 0.6189,
                            },
                            "n_train": 1600,
                            "n_test": 400,
                        },
                    ],
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "code": "model_not_trained",
                    "message": "model is not trained yet, call POST /model/train first",
                    "details": {},
                }
            ]
        }
    }


class PredictionChurn(BaseModel):
    churn: int
    probability_stay: float
    probability_churn: float


class PredictionResponseChurn(BaseModel):
    n_objects: int
    predictions: list[PredictionChurn]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "n_objects": 2,
                    "predictions": [
                        {
                            "churn": 0,
                            "probability_stay": 0.8416,
                            "probability_churn": 0.1584,
                        },
                        {
                            "churn": 1,
                            "probability_stay": 0.4213,
                            "probability_churn": 0.5787,
                        },
                    ],
                }
            ]
        }
    }
