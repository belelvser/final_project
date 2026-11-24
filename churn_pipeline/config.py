"""Конфигурация и явный контракт признаков из ДЗ №5."""
import copy
import json
from pathlib import Path

FEATURES = [
    "cdr_events_30d", "traffic_mb_30d", "voice_minutes_30d",
    "total_charge_30d", "roaming_events_30d", "active_days_30d",
    "closed_tickets_30d", "quality_tickets_30d",
    "avg_latency_ms_30d", "avg_error_rate_30d",
]


def load_config(path, mode=None, output=None):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    config = copy.deepcopy(config)
    config["mode"] = mode or config["mode"]
    if config["mode"] not in ("demo", "full"):
        raise ValueError("Режим должен быть demo или full")
    features = config["features"]
    if (not features or len(features) != len(set(features))
            or not set(features).issubset(FEATURES)):
        raise ValueError("Разрешены только уникальные признаки из ДЗ №5; метка/ID/будущее запрещены")
    if config["target"] != "churn_proxy" or config["id_column"] != "subscriber_key":
        raise ValueError("Этот набор имеет контракт subscriber_key / churn_proxy")
    if config["split"]["strategy"] not in ("preserved", "stratified"):
        raise ValueError("split.strategy: preserved или stratified")
    test, val = config["split"]["test_size"], config["split"]["validation_size"]
    if not (0 < test < 1 and 0 < val < 1 and test + val < 1):
        raise ValueError("Доли validation и test должны быть положительными и в сумме < 1")
    if config["preprocessing"]["imputation"] not in ("median", "mean"):
        raise ValueError("Поддерживаются median и mean для импутации по train")
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed должен быть целым числом от 0 до 2**32-1")
    thresholds = config["evaluation"]["thresholds"]
    if not thresholds or any(not 0 < t < 1 for t in thresholds):
        raise ValueError("Пороги должны находиться между 0 и 1")
    config["evaluation"]["thresholds"] = sorted(set(thresholds))
    if output is not None:
        config["artifacts"]["output_dir"] = str(output)
    return config, path.parent


def resolve_path(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
