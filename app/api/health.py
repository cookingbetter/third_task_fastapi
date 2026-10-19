from fastapi import APIRouter

from app.ml.dataset import dataset_state
from app.ml.model import get_artifact
from app.schemas import HealthStatus

router = APIRouter()


@router.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@router.get("/health", response_model=HealthStatus)
def health():
    # всегда 200: необученная модель — рабочее состояние, а не отказ сервиса
    state = dataset_state()
    return HealthStatus(
        status="ok",
        model_available=get_artifact() is not None,
        dataset_available=state["available"],
        dataset_loaded=state["loaded"],
        n_rows=state["n_rows"],
    )
