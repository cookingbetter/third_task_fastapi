import pandas as pd

from preprocessing import (
    FEATURE_ORDER,
    TARGET,
    TEST_SIZE,
    class_distribution,
    features_to_frame,
    handle_missing,
    prepare_data,
    split_train_test,
)


def row(**overrides) -> dict:
    base = {
        "monthly_fee": 10.0,
        "usage_hours": 20.0,
        "support_requests": 1,
        "account_age_months": 12,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
        "churn": 0,
    }
    return {**base, **overrides}


def test_handle_missing_fills_numeric_with_median():
    df = pd.DataFrame(
        [
            row(monthly_fee=10.0),
            row(monthly_fee=20.0),
            row(monthly_fee=30.0),
            row(monthly_fee=None),
        ]
    )

    result = handle_missing(df)

    assert result["monthly_fee"].tolist() == [10.0, 20.0, 30.0, 20.0]


def test_handle_missing_fills_categorical_with_mode():
    df = pd.DataFrame(
        [
            row(region="europe"),
            row(region="europe"),
            row(region="asia"),
            row(region=None),
        ]
    )

    result = handle_missing(df)

    assert result["region"].tolist() == ["europe", "europe", "asia", "europe"]


def test_handle_missing_drops_rows_without_target():
    df = pd.DataFrame([row(churn=0), row(churn=None), row(churn=1)])

    result = handle_missing(df)

    assert len(result) == 2
    assert result[TARGET].notna().all()


def test_handle_missing_does_not_change_input():
    df = pd.DataFrame([row(monthly_fee=10.0), row(monthly_fee=None)])

    handle_missing(df)

    assert df["monthly_fee"].isna().sum() == 1


def test_prepare_data_returns_features_in_fixed_order():
    df = pd.DataFrame([row(), row(churn=1)])

    X, y = prepare_data(df)

    assert list(X.columns) == FEATURE_ORDER
    assert y.name == TARGET
    assert len(X) == len(y) == 2


def test_prepare_data_ignores_unknown_columns():
    df = pd.DataFrame([row(extra_column="x"), row(churn=1, extra_column="y")])

    X, _ = prepare_data(df)

    assert "extra_column" not in X.columns


def test_features_to_frame_keeps_feature_order():
    shuffled = {name: 1 for name in reversed(FEATURE_ORDER)}

    frame = features_to_frame([shuffled])

    assert list(frame.columns) == FEATURE_ORDER


def test_split_train_test_is_reproducible(synthetic_df):
    X, y = prepare_data(synthetic_df)

    first_train, first_test, _, _ = split_train_test(X, y)
    second_train, second_test, _, _ = split_train_test(X, y)

    assert first_train.index.tolist() == second_train.index.tolist()
    assert first_test.index.tolist() == second_test.index.tolist()


def test_split_train_test_keeps_test_share_and_balance(synthetic_df):
    X, y = prepare_data(synthetic_df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert len(X_test) == round(len(X) * TEST_SIZE)
    assert len(X_train) + len(X_test) == len(X)
    # в synthetic_df классы делятся ровно пополам, стратификация обязана это сохранить
    assert class_distribution(y_test) == {0: len(y_test) // 2, 1: len(y_test) // 2}
    assert class_distribution(y_train) == {0: len(y_train) // 2, 1: len(y_train) // 2}


def test_class_distribution_returns_int_keys():
    y = pd.Series([0.0, 1.0, 1.0])

    assert class_distribution(y) == {0: 1, 1: 2}
