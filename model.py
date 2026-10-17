import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"

MODEL_TYPES = {
    "logreg": (LogisticRegression, {"max_iter": 1000, "random_state": RANDOM_STATE}),
    "random_forest": (RandomForestClassifier, {"random_state": RANDOM_STATE}),
}


class InvalidTrainingConfig(ValueError):
    pass


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    trained_at: datetime
    metrics: dict
    model_type: str = "logreg"
    hyperparameters: dict = field(default_factory=dict)


_artifact: ModelArtifact | None = None


def ensure_finite_hyperparameters(value, name: str) -> None:
    # inf и nan сериализуются в json как null, то есть молча искажают ответ
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidTrainingConfig(
            f"hyperparameter '{name}' must be a finite number, got {value}"
        )
    if isinstance(value, dict):
        for key, item in value.items():
            ensure_finite_hyperparameters(item, f"{name}.{key}" if name else str(key))
    if isinstance(value, list):
        for position, item in enumerate(value):
            ensure_finite_hyperparameters(item, f"{name}[{position}]")


def build_classifier(
    model_type: str,
    hyperparameters: dict | None = None,
) -> tuple[ClassifierMixin, dict]:
    if model_type not in MODEL_TYPES:
        available = ", ".join(sorted(MODEL_TYPES))
        raise InvalidTrainingConfig(
            f"unknown model_type '{model_type}', available: {available}"
        )

    ensure_finite_hyperparameters(hyperparameters or {}, "")

    classifier_cls, defaults = MODEL_TYPES[model_type]
    applied = {**defaults, **(hyperparameters or {})}
    try:
        return classifier_cls(**applied), applied
    except TypeError:
        allowed = ", ".join(sorted(classifier_cls().get_params()))
        unknown = ", ".join(
            sorted(set(applied) - set(classifier_cls().get_params())) or ["-"]
        )
        raise InvalidTrainingConfig(
            f"unknown hyperparameters for '{model_type}': {unknown}; "
            f"allowed: {allowed}"
        )


def build_pipeline(classifier: ClassifierMixin | None = None) -> Pipeline:
    if classifier is None:
        classifier, _ = build_classifier("logreg")
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
            ("classifier", classifier),
        ]
    )


def train_churn_model(
    df: pd.DataFrame,
    model_type: str = "logreg",
    hyperparameters: dict | None = None,
) -> tuple[Pipeline, dict, dict]:
    classifier, applied = build_classifier(model_type, hyperparameters)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    pipeline = build_pipeline(classifier)
    try:
        pipeline.fit(X_train, y_train)
    except (TypeError, ValueError) as error:
        raise InvalidTrainingConfig(
            f"could not train '{model_type}' with given hyperparameters: {error}"
        )
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipeline, metrics, applied


def trained_categories(pipeline: Pipeline) -> dict[str, list[str]]:
    try:
        encoder = pipeline.named_steps["preprocessor"].named_transformers_[
            "categorical"
        ]
        # имена берём у энкодера: он мог быть обучен на другом наборе признаков
        names = encoder.feature_names_in_
        return {
            str(name): [str(value) for value in values]
            for name, values in zip(names, encoder.categories_)
        }
    except (AttributeError, KeyError):
        # артефакт мог быть сохранён другой версией кода
        return {}


def save_churn_model(artifact: ModelArtifact, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_churn_model(path: Path = MODEL_PATH) -> ModelArtifact | None:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        logger.warning(
            "could not load model from %s, treating as not trained",
            path,
            exc_info=True,
        )
        return None


def set_artifact(artifact: ModelArtifact) -> None:
    global _artifact
    _artifact = artifact


def get_artifact() -> ModelArtifact | None:
    return _artifact


def get_model() -> Pipeline | None:
    return _artifact.pipeline if _artifact else None
