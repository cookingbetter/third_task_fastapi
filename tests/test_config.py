from datetime import datetime, timezone

from app.core.config import BASE_DIR, DATASET_PATH, HISTORY_PATH, MODEL_PATH
from app.ml.history import append_record, load_history
from app.ml.model import ModelArtifact, load_churn_model, save_churn_model, train_churn_model


def test_base_dir_points_at_the_project_root():
    # BASE_DIR считается через parents[2], и сдвиг на уровень молча увёл бы
    # и датасет, и каталог модели в сторону
    assert (BASE_DIR / "app" / "core" / "config.py").is_file()
    assert (BASE_DIR / "pytest.ini").is_file()


def test_paths_are_built_from_the_project_root():
    assert DATASET_PATH == BASE_DIR / "data" / "churn_dataset.csv"
    assert DATASET_PATH.is_file()
    assert MODEL_PATH == BASE_DIR / "models" / "churn_model.joblib"
    assert HISTORY_PATH == BASE_DIR / "models" / "training_history.json"


def test_model_is_saved_to_the_configured_path(synthetic_df, isolated_storage):
    # путь не передаём: проверяем, что функция берёт его в момент вызова,
    # иначе каталог тома в контейнере перестал бы совпадать с местом записи
    pipeline, metrics, _ = train_churn_model(synthetic_df, "logreg", {})
    artifact = ModelArtifact(
        pipeline=pipeline, trained_at=datetime.now(timezone.utc), metrics=metrics
    )

    save_churn_model(artifact)

    assert isolated_storage["model"].is_file()
    assert load_churn_model() is not None


def test_history_is_written_to_the_configured_path(isolated_storage):
    append_record({"model_type": "logreg"})

    assert isolated_storage["history"].is_file()
    assert load_history() == [{"model_type": "logreg"}]
