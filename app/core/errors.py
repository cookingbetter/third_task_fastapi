DATASET_NOT_FOUND = "dataset_not_found"
EMPTY_DATASET = "empty_dataset"
EMPTY_PAYLOAD = "empty_payload"
MODEL_NOT_TRAINED = "model_not_trained"
FEATURE_MISMATCH = "feature_mismatch"
INCOMPATIBLE_MODEL = "incompatible_model"
INVALID_TARGET = "invalid_target"
VALIDATION_ERROR = "validation_error"
NON_FINITE_VALUE = "non_finite_value"
INVALID_TRAINING_CONFIG = "invalid_training_config"
DATA_PREPARATION_FAILED = "data_preparation_failed"
PREDICTION_FAILED = "prediction_failed"
INTERNAL_ERROR = "internal_error"

# коды для ошибок, которые поднимает сам fastapi, а не наш код
STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    422: VALIDATION_ERROR,
    500: INTERNAL_ERROR,
}


class InvalidTrainingConfig(ValueError):
    """Тип модели или гиперпараметры, с которыми обучение невозможно."""


class ServiceError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def error_payload(code: str, message: str, details: dict | None = None) -> dict:
    return {"code": code, "message": message, "details": details or {}}
