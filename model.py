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

_trained_model: Pipeline | None = None


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


def set_model(pipeline: Pipeline) -> None:
    global _trained_model
    _trained_model = pipeline


def get_model() -> Pipeline | None:
    return _trained_model
