# Архитектура конвейера

```mermaid
flowchart TD
    subgraph H[Подготовка данных в Spark]
        R[TDG S: CDR, Tickets, Telemetry]
        Q[Профилирование и преобразование в Parquet]
        F[Очистка и 10 агрегатов за 31.10–29.11.2025]
        T[Будущая CDR-неактивность за 30.11–29.12.2025]
        V[Витрина: 186463 абонента, churn_proxy, split]
        R --> Q --> F --> V
        R --> T --> V
    end
    subgraph D[Слой входных данных итогового проекта]
        Z[data/full.parquet: 186463 строк, 6,3 МБ]
        S[data/sample.parquet: 6000 строк]
        L[data.load_data: размер, SHA256, чтение Parquet]
        C[data.validate_data: схема, ключи, числа, метка, split]
        X[Явный выбор разрешённых признаков]
        V --> Z
        Z -. prepare_sample: стратифицированная выборка .-> S
        Z --> L
        S --> L
        L --> C --> X
    end
    subgraph M[ML-слой]
        P[preprocess: сохранённое разделение абонентов 60/20/20]
        I[Pipeline: SimpleImputer.fit только на train]
        B[train_model: XGBoost.fit только на train]
        U[evaluate: выбор порога F1 по validation]
        E[Оценка test: ROC-AUC, AP, F1, precision, recall, матрица ошибок]
        X --> P --> I --> B --> U --> E
    end
    subgraph A[Слой хранения и управления]
        J[config.json: пути, признаки, режим, seed, split, параметры]
        K[run_pipeline: lock, fingerprint, проверка кэша]
        G[logging: консоль и ограниченный по размеру pipeline.log]
        W[save_artifacts: временная папка, SHA256 manifest, атомарный rename]
        O[artifacts*/runs/id: модель, UBJSON, imputer, метрики, split, прогнозы, конфиг, происхождение]
        N[latest.json: атомарный указатель на завершённый запуск]
        J --> K
        J -.-> L
        J -.-> P
        J -.-> B
        J -.-> U
        K --> L
        K --> G
        K -. при совпадении входа, кода, настроек и среды — проверка файлов .-> O
        E --> W --> O --> N
    end
```

В Docker выполняется тот же `main.py`; `/outputs` связывается с папкой Windows.
После завершения контейнер сохраняется; повторный запуск — `docker start -a <имя>`.
Spark-слой формирует витрину признаков и меток. Точка входа `main.py` начинает
выполнение с чтения `data/full.parquet` или `data/sample.parquet`.
Окна признаков и ограничения доступности поздних данных приведены в README.

Статистическая предобработка расположена **после разделения**: медианы не вычисляются
по validation/test. В режиме `stratified` используется новое разбиение с seed из
конфигурации; вариант `preserved` использует группы из столбца `split`.
