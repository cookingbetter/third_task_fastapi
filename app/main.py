import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api import dataset, health, model, predict
from app.core.errors import (
    EMPTY_PAYLOAD,
    INTERNAL_ERROR,
    INVALID_TRAINING_CONFIG,
    NON_FINITE_VALUE,
    STATUS_CODES,
    VALIDATION_ERROR,
    InvalidTrainingConfig,
    ServiceError,
    error_payload,
)
from app.core.logging_config import configure_logging
from app.ml.model import load_churn_model, set_artifact

logger = logging.getLogger(__name__)

# ошибка в самом запросе — рядовое событие, а неготовность сервиса стоит заметить
CLIENT_ERROR_CODES = {
    VALIDATION_ERROR,
    EMPTY_PAYLOAD,
    NON_FINITE_VALUE,
    INVALID_TRAINING_CONFIG,
    STATUS_CODES[400],
    STATUS_CODES[404],
    STATUS_CODES[405],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("churn service starting")
    artifact = load_churn_model()
    if artifact is not None:
        set_artifact(artifact)
        logger.info(
            "model restored from disk: model_type=%s trained_at=%s",
            getattr(artifact, "model_type", None),
            artifact.trained_at,
        )
    else:
        logger.info("no stored model found, service starts untrained")
    yield
    logger.info("churn service stopped")


app = FastAPI(title="ML Churn Service", lifespan=lifespan)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(dataset.router)
app.include_router(model.router)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def log_error(request: Request, status_code: int, code: str, message: str) -> None:
    # клиентские ошибки идут как info, отказы сервиса как warning
    level = logging.INFO if status_code < 500 and code in CLIENT_ERROR_CODES else logging.WARNING
    logger.log(
        level,
        "%s %s -> %s %s: %s",
        request.method,
        request.url.path,
        status_code,
        code,
        message,
    )


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    log_error(request, exc.status_code, exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(exc.code, exc.message, exc.details),
    )


@app.exception_handler(InvalidTrainingConfig)
async def training_config_error_handler(request: Request, exc: InvalidTrainingConfig):
    log_error(request, 422, INVALID_TRAINING_CONFIG, str(exc))
    return JSONResponse(
        status_code=422,
        content=error_payload(INVALID_TRAINING_CONFIG, str(exc)),
    )


def compact_errors(errors) -> list[dict]:
    # присланное значение не возвращаем: на большом батче оно раздувает ответ
    return [
        {
            "type": error.get("type"),
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg"),
        }
        for error in errors
    ]


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    log_error(
        request, 422, VALIDATION_ERROR, f"{len(exc.errors())} validation error(s)"
    )
    return JSONResponse(
        status_code=422,
        content=error_payload(
            VALIDATION_ERROR,
            "request does not match the expected schema",
            {"errors": json_safe(jsonable_encoder(compact_errors(exc.errors())))},
        ),
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    # сюда попадает то, что поднимает сам fastapi: неизвестный путь, чужой метод
    code = STATUS_CODES.get(exc.status_code, "http_error")
    log_error(request, exc.status_code, code, str(exc.detail))
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(code, str(exc.detail)),
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled error while processing %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content=error_payload(INTERNAL_ERROR, "internal server error"),
    )
