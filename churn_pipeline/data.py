"""Чтение готовой Parquet-витрины из репозитория с проверкой целостности."""
import hashlib
import io
import logging

import numpy as np
import pandas as pd

from .config import resolve_path

LOG = logging.getLogger(__name__)


def load_data(config, root):
    source = config["data"][config["mode"]]
    path = resolve_path(root, source["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Нет входных данных: {path}. Проверьте data в config.json")
    if path.stat().st_size > source["max_bytes"]:
        raise ValueError("Файл превышает разрешённый размер компактной витрины")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source["sha256"]:
        raise ValueError("SHA256 входных данных не совпадает с config.json; проверьте происхождение файла")
    frame = pd.read_parquet(io.BytesIO(payload))
    LOG.info("Загрузка: %s, строк=%d, SHA256=%s", path.name, len(frame), digest)
    return frame, digest


def validate_data(frame, config):
    """Проверки не обучаются на данных; статистики импутации здесь не считаются."""
    features, target, key = config["features"], config["target"], config["id_column"]
    required = [key, target, *features]
    if config["split"]["strategy"] == "preserved":
        required.append("split")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные столбцы: {sorted(missing)}")
    frame = frame[required].copy()
    frame[key] = frame[key].astype("string").str.strip()
    if frame[key].isna().any() or frame[key].eq("").any() or frame[key].duplicated().any():
        raise ValueError("subscriber_key должен быть непустым и уникальным; пересечение абонентов запрещено")
    if frame[target].isna().any() or not set(frame[target].unique()) == {0, 1}:
        raise ValueError("Целевая переменная должна содержать оба класса: 0 и 1")
    frame[target] = frame[target].astype("int64")
    for col in features:
        try:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Признак {col} содержит нечисловые значения") from exc
        if np.isinf(frame[col].to_numpy()).any():
            raise ValueError(f"Бесконечное значение в {col}")
        if frame[col].lt(0).any():
            raise ValueError(f"Отрицательное значение в {col}")
    if "cdr_events_30d" in frame and not frame["cdr_events_30d"].gt(0).all():
        raise ValueError("Когорта должна иметь хотя бы одно CDR-событие в окне признаков")
    if "active_days_30d" in frame and frame["active_days_30d"].gt(30).any():
        raise ValueError("active_days_30d превышает 30")
    if "split" in frame and (frame["split"].isna().any()
                              or set(frame["split"].unique()) != {"train", "validation", "test"}):
        raise ValueError("split должен содержать train, validation и test без пропусков")
    return frame
