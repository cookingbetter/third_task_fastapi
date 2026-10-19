import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.ml.history import HISTORY_LIMIT, append_record, load_history


@pytest.fixture
def history_path(tmp_path):
    return tmp_path / "training_history.json"


def record(seq: int) -> dict:
    return {
        "trained_at": "2026-10-18T10:00:00+00:00",
        "model_type": "logreg",
        "hyperparameters": {"seq": seq},
        "metrics": {"accuracy": 0.8, "f1": 0.5, "roc_auc": 0.7},
        "n_train": 40,
        "n_test": 10,
    }


def seqs(history_path) -> list[int]:
    return [item["hyperparameters"]["seq"] for item in load_history(history_path)]


def test_load_history_returns_empty_list_when_file_is_missing(history_path):
    assert load_history(history_path) == []


def test_append_record_creates_file_and_keeps_order(history_path):
    append_record(record(1), history_path)
    append_record(record(2), history_path)

    assert history_path.exists()
    assert seqs(history_path) == [1, 2]


def test_append_record_creates_missing_directory(tmp_path):
    path = tmp_path / "models" / "training_history.json"

    append_record(record(1), path)

    assert seqs(path) == [1]


def test_append_record_leaves_no_temporary_file(history_path):
    append_record(record(1), history_path)

    assert list(history_path.parent.glob("*.tmp")) == []


def test_append_record_respects_limit(history_path):
    total = HISTORY_LIMIT + 30
    for seq in range(total):
        append_record(record(seq), history_path)

    assert seqs(history_path) == list(range(total - HISTORY_LIMIT, total))


@pytest.mark.parametrize(
    "content",
    ["not json at all", '{"model_type": "logreg"}', ""],
)
def test_load_history_tolerates_broken_file(history_path, content):
    history_path.write_text(content)

    assert load_history(history_path) == []


def test_parallel_appends_do_not_lose_records(history_path):
    # регрессия дня 11: без блокировки одновременные записи затирали друг друга
    total = 20
    with ThreadPoolExecutor(max_workers=total) as pool:
        list(pool.map(lambda seq: append_record(record(seq), history_path), range(total)))

    assert sorted(seqs(history_path)) == list(range(total))


def test_history_file_stays_valid_json(history_path):
    append_record(record(1), history_path)

    assert isinstance(json.loads(history_path.read_text(encoding="utf-8")), list)
