from pathlib import Path

# app/core/config.py -> app/core -> app -> корень проекта
BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "data" / "churn_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
HISTORY_PATH = MODELS_DIR / "training_history.json"

# сколько записей истории обучений храним и сколько максимум отдаёт /model/metrics
HISTORY_LIMIT = 100

TEST_SIZE = 0.2
RANDOM_STATE = 42

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DEFAULT_LOG_LEVEL = "INFO"
