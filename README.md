# third_task_fastapi

Сервис предсказания оттока клиентов (churn) на FastAPI и scikit-learn.

## Запуск локально

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --reload
```

Документация доступна на `http://127.0.0.1:8000/docs`, состояние сервиса — на `/health`.

## Тесты

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

## Запуск в Docker

```bash
docker build -t churn-service .
docker run -d -p 8000:8000 -v churn-models:/app/models --name churn churn-service
```

Датасет копируется внутрь образа, а каталог `models/` вынесен в том `churn-models`, поэтому
обученная модель и история обучений сохраняются между перезапусками контейнера.

Полезные команды:

```bash
curl http://127.0.0.1:8000/health
docker logs -f churn
docker inspect --format '{{.State.Health.Status}}' churn
```

Уровень логирования задаётся переменной окружения `LOG_LEVEL` (по умолчанию `INFO`):

```bash
docker run -d -p 8000:8000 -e LOG_LEVEL=DEBUG --name churn churn-service
```
