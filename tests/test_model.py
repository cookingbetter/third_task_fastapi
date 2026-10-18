import math
from datetime import datetime, timezone

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from model import (
    InvalidTrainingConfig,
    ModelArtifact,
    build_classifier,
    load_churn_model,
    save_churn_model,
    train_churn_model,
    trained_categories,
)
from preprocessing import CATEGORICAL_FEATURES, FEATURE_ORDER, features_to_frame

LOYAL_FEATURES = {
    "monthly_fee": 10.0,
    "usage_hours": 42.0,
    "support_requests": 0,
    "account_age_months": 26,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1,
}


def test_build_classifier_applies_defaults():
    classifier, applied = build_classifier("logreg")

    assert isinstance(classifier, LogisticRegression)
    assert applied == {"max_iter": 1000, "random_state": 42}


def test_build_classifier_merges_hyperparameters():
    classifier, applied = build_classifier("random_forest", {"n_estimators": 20})

    assert isinstance(classifier, RandomForestClassifier)
    assert applied == {"random_state": 42, "n_estimators": 20}


@pytest.mark.parametrize(
    "model_type, hyperparameters",
    [
        ("xgboost", None),
        ("logreg", {"depth": 3}),
        ("logreg", {"C": math.inf}),
        ("random_forest", {"n_estimators": float("nan")}),
    ],
)
def test_build_classifier_rejects_bad_config(model_type, hyperparameters):
    with pytest.raises(InvalidTrainingConfig):
        build_classifier(model_type, hyperparameters)


def test_train_churn_model_returns_pipeline_and_metrics(synthetic_df):
    pipeline, metrics, applied = train_churn_model(synthetic_df)

    assert list(pipeline.feature_names_in_) == FEATURE_ORDER
    assert [int(value) for value in pipeline.classes_] == [0, 1]
    assert applied == {"max_iter": 1000, "random_state": 42}
    assert metrics["n_train"] + metrics["n_test"] == len(synthetic_df)
    for name in ("accuracy", "f1", "roc_auc"):
        assert 0.0 <= metrics[name] <= 1.0


def test_train_churn_model_learns_separable_data(synthetic_df):
    # классы в synthetic_df разделимы, слабый результат означал бы поломку конвейера
    _, metrics, _ = train_churn_model(synthetic_df)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_train_churn_model_is_reproducible(synthetic_df):
    _, first, _ = train_churn_model(synthetic_df)
    _, second, _ = train_churn_model(synthetic_df)

    assert first == second


def test_train_churn_model_rejects_unknown_model_type(synthetic_df):
    with pytest.raises(InvalidTrainingConfig):
        train_churn_model(synthetic_df, "xgboost")


def test_save_and_load_model_round_trip(synthetic_df, tmp_path):
    pipeline, metrics, applied = train_churn_model(
        synthetic_df, "random_forest", {"n_estimators": 20}
    )
    artifact = ModelArtifact(
        pipeline=pipeline,
        trained_at=datetime.now(timezone.utc),
        metrics=metrics,
        model_type="random_forest",
        hyperparameters=applied,
    )
    path = tmp_path / "nested" / "churn_model.joblib"

    save_churn_model(artifact, path)
    loaded = load_churn_model(path)

    assert loaded is not None
    assert loaded.model_type == "random_forest"
    assert loaded.hyperparameters == applied
    assert loaded.metrics == metrics
    assert loaded.trained_at == artifact.trained_at

    X = features_to_frame([LOYAL_FEATURES])
    assert loaded.pipeline.predict(X).tolist() == pipeline.predict(X).tolist()
    assert loaded.pipeline.predict_proba(X).tolist() == pipeline.predict_proba(X).tolist()


def test_load_model_returns_none_when_file_is_missing(tmp_path):
    assert load_churn_model(tmp_path / "nothing.joblib") is None


def test_load_model_returns_none_when_file_is_broken(tmp_path):
    path = tmp_path / "churn_model.joblib"
    path.write_text("not a joblib artifact")

    assert load_churn_model(path) is None


def test_trained_categories_lists_values_per_feature(synthetic_df):
    pipeline, _, _ = train_churn_model(synthetic_df)

    categories = trained_categories(pipeline)

    assert sorted(categories) == sorted(CATEGORICAL_FEATURES)
    assert categories["region"] == ["africa", "europe"]


def test_trained_categories_tolerates_untrained_pipeline():
    from model import build_pipeline

    assert trained_categories(build_pipeline()) == {}
