import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=[TARGET]).copy()
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = handle_missing(df)
    X = df[FEATURE_ORDER]
    y = df[TARGET]
    return X, y


def features_to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=FEATURE_ORDER)


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def class_distribution(y: pd.Series) -> dict[int, int]:
    return {int(k): int(v) for k, v in y.value_counts().sort_index().items()}
