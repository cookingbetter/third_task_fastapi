from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    prepare_data,
    split_train_test,
)

MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    trained_at: datetime
    metrics: dict


_artifact: ModelArtifact | None = None


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def train_churn_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipeline, metrics


def save_churn_model(artifact: ModelArtifact, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_churn_model(path: Path = MODEL_PATH) -> ModelArtifact | None:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def set_artifact(artifact: ModelArtifact) -> None:
    global _artifact
    _artifact = artifact


def get_artifact() -> ModelArtifact | None:
    return _artifact


def get_model() -> Pipeline | None:
    return _artifact.pipeline if _artifact else None
