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
    PREDICTION_FAILED,
    VALIDATION_ERROR,
)
from app.ml.preprocessing import FEATURE_ORDER, TARGET
from app.schemas import ErrorResponse

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
        "description": "Объект передаётся как есть, ответ всё равно содержит "
        "список из одного прогноза.",
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

DATASET_LOAD_RESPONSE = error_responses(
    "Файл датасета не читается", {"dataset_load_failed": DATASET_LOAD_EXAMPLE}
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
    400: EMPTY_DATASET_RESPONSE,
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
