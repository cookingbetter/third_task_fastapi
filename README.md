# Churn service

HTTP-сервис на FastAPI и scikit-learn, который предсказывает отток клиентов (churn).
По описанию клиента — тарифу, активности, обращениям в поддержку — сервис отвечает,
уйдёт клиент или останется, и с какой вероятностью.

Что умеет сервис:

- обучать модель на размеченном датасете (`POST /model/train`), логистическую регрессию
  или случайный лес, с настраиваемыми гиперпараметрами;
- сохранять обученную модель на диск и поднимать её при старте, поэтому после
  перезапуска обучать заново не нужно;
- предсказывать отток для одного клиента или батча (`POST /predict`);
- показывать состояние модели, историю обучений и схему признаков
  (`/model/status`, `/model/metrics`, `/model/schema`);
- отдавать сведения о датасете (`/dataset/info`, `/dataset/preview`, `/dataset/split-info`);
- сообщать о собственной готовности (`/health`).

## Формат датасета

Данные лежат в `data/churn_dataset.csv`: 2000 строк, 10 столбцов — девять признаков и
целевая переменная `churn`, пропусков нет. Файл читается один раз при первом обращении
и проверяется строка за строкой схемой `DatasetRowChurn`: обязательны все девять
признаков и `churn`, значения должны приводиться к указанному типу. Лишние столбцы
игнорируются, а пропущенный столбец или нечисловое значение в числовом поле дают
ошибку с кодом `data_preparation_failed`.

В таблице — типы столбцов и значения, которые встречаются в текущем датасете.

| столбец | тип | значения в датасете | смысл |
| --- | --- | --- | --- |
| `monthly_fee` | float | 9.99 … 49.99 | ежемесячный платёж |
| `usage_hours` | float | 0.0 … 52.07 | часы использования сервиса |
| `support_requests` | int | 0 … 4 | число обращений в поддержку |
| `account_age_months` | int | 1 … 39 | возраст аккаунта в месяцах |
| `failed_payments` | int | 0 … 2 | неудавшиеся списания |
| `region` | str | `europe`, `asia`, `america`, `africa` | регион клиента |
| `device_type` | str | `mobile`, `desktop`, `tablet` | основное устройство |
| `payment_method` | str | `card`, `paypal`, `crypto` | способ оплаты |
| `autopay_enabled` | int | `0`, `1` | включён ли автоплатёж |
| `churn` | int | `0`, `1` | целевая переменная: 1 — клиент ушёл |

Числовые признаки — пять первых и `autopay_enabled`, они масштабируются
`StandardScaler`; категориальные — `region`, `device_type` и `payment_method`, они
кодируются `OneHotEncoder`. Диапазоны и перечни категорий сервис не
навязывает: в `/predict` принимается любое число и любая строка, а незнакомая
категория кодируется нулями (`handle_unknown="ignore"`) — запрос отработает, но
такому прогнозу доверять не стоит. Допустимые значения, на которых обучена текущая
модель, можно посмотреть в `GET /model/schema`.

Порядок признаков фиксирован (`FEATURE_ORDER`) и запоминается в модели, поэтому
артефакт, обученный на другом наборе, будет отклонён с ошибкой `feature_mismatch`.

Модель сохраняется в `models/churn_model.joblib` через `joblib`, то есть внутри
артефакта лежат ссылки на классы проекта и scikit-learn. Артефакт, сохранённый другой
версией библиотек или прежней структурой модулей, загрузить не удастся: сервис запишет
предупреждение в лог, стартует как необученный, и модель нужно будет обучить заново
запросом `POST /model/train`.

Классы несбалансированы: 1597 строк с `churn=0` и 403 с `churn=1`, то есть 20.15% оттока.
Разбиение на train и test — 80/20 со стратификацией по `churn` и фиксированным
`random_state=42`, так что метрики воспроизводимы.

## Структура проекта

```
app/
  main.py              создание приложения, lifespan, обработчики ошибок
  schemas.py           pydantic-схемы запросов и ответов
  api/
    deps.py            зависимость датасета и общие проверки запросов
    docs.py            примеры запросов и ответов для OpenAPI
    health.py          GET / и GET /health
    dataset.py         роутер /dataset/*
    model.py           роутер /model/*
    predict.py         POST /predict
  core/
    config.py          пути, параметры разбиения, настройки логов
    errors.py          коды ошибок и исключения сервиса
    logging_config.py  настройка логирования
  ml/
    dataset.py         чтение и кеширование датасета
    preprocessing.py   признаки, предобработка, разбиение на train/test
    model.py           pipeline, обучение, сохранение и загрузка артефакта
    history.py         история обучений с блокировкой файла
data/churn_dataset.csv  обучающий датасет
models/                 обученная модель и история обучений
tests/                  pytest-тесты
```

Все пути собраны в `app/core/config.py` и строятся от корня проекта, поэтому сервис
можно запускать из любого каталога.

## Запуск локально

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

Интерактивная документация — `http://127.0.0.1:8000/docs`, состояние сервиса — `/health`.

## Запуск в Docker

```bash
docker build -t churn-service .
docker run -d -p 8000:8000 -v churn-models:/app/models --name churn churn-service
```

Датасет копируется внутрь образа, а каталог `models/` вынесен в том `churn-models`,
поэтому обученная модель и история обучений переживают перезапуск контейнера.
У образа есть `HEALTHCHECK`, который опрашивает `/health`.

Полезные команды:

```bash
curl http://127.0.0.1:8000/health
docker logs -f churn
docker inspect --format '{{.State.Health.Status}}' churn
```

Уровень логирования задаётся переменной окружения `LOG_LEVEL` (по умолчанию `INFO`):

```bash
docker run -d -p 8000:8000 -v churn-models:/app/models -e LOG_LEVEL=WARNING \
  --name churn-debug churn-service
```

## Примеры запросов

### Обучение модели

Тело запроса можно не передавать — тогда обучается логистическая регрессия:

```bash
curl -X POST http://127.0.0.1:8000/model/train
```

```json
{
  "status": "trained",
  "algorithm": "LogisticRegression",
  "model_type": "logreg",
  "hyperparameters": {"max_iter": 1000, "random_state": 42},
  "n_train": 1600,
  "n_test": 400,
  "metrics": {"accuracy": 0.7875, "f1": 0.0449, "roc_auc": 0.6091}
}
```

Другой тип модели и свои гиперпараметры:

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H 'content-type: application/json' \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 8}}'
```

```json
{
  "status": "trained",
  "algorithm": "RandomForestClassifier",
  "model_type": "random_forest",
  "hyperparameters": {"random_state": 42, "n_estimators": 200, "max_depth": 8},
  "n_train": 1600,
  "n_test": 400,
  "metrics": {"accuracy": 0.795, "f1": 0.1458, "roc_auc": 0.6148}
}
```

Доступные типы моделей — `logreg` и `random_forest`; неизвестный тип или неизвестный
гиперпараметр отклоняются с кодом `invalid_training_config`, а допустимые значения
перечисляются в тексте ошибки.

### Предсказание

Один клиент передаётся объектом, несколько — списком:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '[
    {"monthly_fee": 9.99, "usage_hours": 27.92, "support_requests": 1,
     "account_age_months": 14, "failed_payments": 0, "region": "europe",
     "device_type": "desktop", "payment_method": "card", "autopay_enabled": 1},
    {"monthly_fee": 49.99, "usage_hours": 1.5, "support_requests": 4,
     "account_age_months": 1, "failed_payments": 2, "region": "africa",
     "device_type": "mobile", "payment_method": "crypto", "autopay_enabled": 0}
  ]'
```

```json
{
  "n_objects": 2,
  "predictions": [
    {"churn": 0, "probability_stay": 0.8735, "probability_churn": 0.1265},
    {"churn": 1, "probability_stay": 0.486, "probability_churn": 0.514}
  ]
}
```

Вероятности в ответе зависят от того, какая модель обучена последней: здесь показан
ответ случайного леса из примера выше, у логистической регрессии числа будут другими.

Ответ всегда содержит список прогнозов в том же порядке, что и присланные объекты,
даже если был передан один объект. Запрос к необученной модели возвращает 409 с кодом
`model_not_trained`.

### Остальные эндпоинты

| метод и путь | назначение |
| --- | --- |
| `GET /` | краткий ответ, что сервис жив |
| `GET /health` | доступность модели и датасета, число загруженных строк |
| `POST /model/train` | обучение модели |
| `GET /model/status` | обучена ли модель, когда, с какими метриками |
| `GET /model/metrics` | история обучений, фильтр `model_type`, параметр `limit` |
| `GET /model/schema` | порядок и типы признаков, допустимые категории |
| `POST /predict` | предсказание оттока |
| `GET /dataset/preview` | первые `n` строк датасета |
| `GET /dataset/info` | размер датасета и баланс классов |
| `GET /dataset/split-info` | параметры разбиения и баланс в train и test |

## Формат ошибок

Все ошибки возвращаются одной структурой: код для программной обработки, сообщение для
человека и необязательные детали.

Например, лишний признак в теле `/predict` даёт 422. Тело принимается либо объектом,
либо списком, поэтому в деталях видно по ошибке на каждый из двух вариантов:

```json
{
  "code": "validation_error",
  "message": "request does not match the expected schema",
  "details": {
    "errors": [
      {"type": "extra_forbidden", "loc": ["body", "FeatureVectorChurn", "extra"],
       "msg": "Extra inputs are not permitted"},
      {"type": "list_type", "loc": ["body", "list[FeatureVectorChurn]"],
       "msg": "Input should be a valid list"}
    ]
  }
}
```

Присланные значения в ответ не попадают, трассировки не раскрываются — подробности
остаются в логах сервиса. Коды ошибок перечислены в `app/core/errors.py`, а примеры
каждого из них видны в `/docs`.

## Тесты

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

87 тестов: пути из конфигурации, предобработка и обучение без FastAPI, история
обучений вместе с конкурентной записью, интеграционные сценарии через `TestClient`
и обработка ошибок. Тесты не трогают настоящие `models/` — на время прогона
подменяются `MODEL_PATH` и `HISTORY_PATH`, и запись уходит в каталог `tmp_path`.
