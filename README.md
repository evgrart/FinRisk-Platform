# FinRisk Platform

Production-like ML-проект для финтеха: единый сервис оценивает риск дефолта по заявке на кредит и вероятность мошенничества по транзакции.

Проект специально собран так, чтобы показать не только обучение модели, но и инженерное мышление, важное для банковского ML:

- воспроизводимая генерация обезличенных синтетических данных;
- временное разбиение и защита от data leakage;
- сравнение baseline и более сильных моделей;
- калибровка вероятностей и выбор порога под стоимость ошибок;
- интерпретация решения для аналитика;
- FastAPI inference-сервис, Docker, тесты и мониторинг дрейфа.

> В репозитории нет настоящих банковских данных. Все записи генерируются локально и нужны только для демонстрации методологии.

## Зачем этот проект

Вместо игрушечного `fit/predict` здесь есть реалистичный контур принятия решения:

```text
synthetic events
      |
      v
data validation -> feature pipeline -> model training -> calibration
      |                                             |
      +-------------------------------> FastAPI scoring service
                                                    |
                                  score + decision + explanation + model version
```

Основные бизнес-метрики:

- кредитный риск: ROC-AUC, PR-AUC, recall при ограничении на долю одобрений, Brier score;
- антифрод: PR-AUC, recall при фиксированном уровне false positive rate, стоимость ошибок;
- сервис: p95 latency, доля ошибок, стабильность распределения score.

## Структура репозитория

```text
FinRisk-Platform/
├── configs/              # параметры экспериментов
├── data/
│   ├── raw/              # генерируемые исходные CSV, не коммитятся
│   └── processed/        # подготовленные датасеты, не коммитятся
├── docs/                 # постановка задачи и история решений
├── notebooks/            # EDA и разбор результатов
├── src/finrisk/
│   ├── data/             # генерация и валидация данных
│   ├── features/         # feature engineering
│   ├── models/           # обучение и оценка
│   ├── service/          # API инференса
│   └── monitoring/       # контроль качества и дрейфа
├── tests/
└── pyproject.toml
```

## Быстрый старт

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,analysis]"
python -m finrisk.data.generate --seed 42 --applications 50000 --transactions 150000
python -m finrisk.data.validate --credit data/raw/credit_applications.csv --transactions data/raw/transactions.csv
python -m finrisk.models.train_credit --data data/raw/credit_applications.csv
python -m finrisk.models.train_fraud --data data/raw/transactions.csv
pytest
```

## Tech stack

| Слой | Технологии | Как используется |
| --- | --- | --- |
| Язык и данные | Python 3.11+, NumPy, Pandas | генерация событий, очистка и feature engineering |
| ML | scikit-learn | Logistic Regression, Random Forest, temporal validation, calibration |
| Risk policy | NumPy, собственные cost-модули | выбор порогов и стоимость false positive/false negative |
| API | FastAPI, Pydantic, Uvicorn | типизированный scoring API и Swagger |
| Explainability | прозрачные reason codes, Model Card | причины для ручной проверки и ограничения модели |
| Monitoring | PSI, Pandas | контроль drift между reference и current snapshot |
| Engineering | pytest, Ruff, GitHub Actions | тесты, lint и CI на каждый push |
| Deployment | Docker, Docker Compose | воспроизводимый запуск inference-сервиса |

Важно: в проекте используются только те технологии, которые реально задействованы
в коде. PostgreSQL и PyTorch можно добавить отдельными этапами, если появится
необходимость в online feature store или deep-learning baseline.

После команды генерации данные появятся в `data/raw/`:

- `credit_applications.csv` — 50 000 заявок с бинарной целью `default_90d`;
- `transactions.csv` — 150 000 транзакций с бинарной целью `is_fraud`.

## План разработки и коммитов

Каждый пункт — отдельный осмысленный коммит:

1. `chore: scaffold FinRisk project and reproducible synthetic data` — завершено;
2. `feat: add data validation and temporal split` — завершено;
3. `feat: train calibrated credit risk models` — завершено;
4. `feat: train transaction fraud detector and cost-sensitive threshold` — завершено;
5. `feat: expose credit and fraud scoring through FastAPI` — завершено;
6. `feat: add explanations and model card` — завершено;
7. `feat: add drift monitoring and quality report` — завершено;
8. `ci: add tests, lint and Docker deployment` — завершено.

Финальный результат должен позволять открыть Swagger, отправить JSON-заявку или транзакцию и получить не только score, но и понятное решение: `approve/review/reject` либо `allow/review/block`.

После обучения моделей сервис запускается командой:

```powershell
uvicorn finrisk.service.app:app --reload
```

Swagger будет доступен по адресу `http://localhost:8000/docs`.

Пример ответа API:

```json
{
  "risk_score": 0.1842,
  "decision": "review",
  "threshold": 0.2175,
  "model_version": "random_forest",
  "reasons": ["высокая долговая нагрузка"]
}
```

Для контейнерного запуска после обучения моделей:

```powershell
docker compose up --build
```

Каталог `artifacts/` монтируется в контейнер только для чтения, поэтому API
не может случайно перезаписать модель во время работы.

## Что положить в резюме после завершения

> Разработал end-to-end платформу кредитного и транзакционного risk-scoring: leak-free temporal validation, калибровка вероятностей, cost-sensitive thresholding, explainable FastAPI inference, Docker и мониторинг drift; воспроизводимость обеспечена seed/versioned configs.

Метрики в эту строку нужно подставить только после реального запуска экспериментов.
