import fcntl
import json
import logging
from contextlib import contextmanager
from pathlib import Path

from app.core.config import HISTORY_LIMIT, HISTORY_PATH

logger = logging.getLogger(__name__)


def load_history(path: Path | None = None) -> list[dict]:
    # путь читается при вызове, а не фиксируется значением по умолчанию:
    # иначе подменить его в тестах можно было бы только заменой самой функции
    path = path or HISTORY_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("could not read history from %s", path, exc_info=True)
        return []
    return data if isinstance(data, list) else []


@contextmanager
def locked(path: Path):
    # блокировка живёт в отдельном файле: сам файл истории заменяется целиком,
    # и дескриптор на нём после replace указывал бы уже на удалённую версию
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def append_record(record: dict, path: Path | None = None) -> None:
    path = path or HISTORY_PATH
    # без блокировки одновременные обучения читают одно состояние
    # и затирают записи друг друга
    with locked(path):
        history = load_history(path)[-(HISTORY_LIMIT - 1) :] + [record]
        # пишем через временный файл: сбой на полпути не должен обнулить историю
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(path)
