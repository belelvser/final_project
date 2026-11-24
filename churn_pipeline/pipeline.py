"""Единый запуск, диагностика, кэш и повторная проверка сохранённой модели."""
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
import pandas as pd

from .artifacts import (
    output_lock, run_identity, save_artifacts, set_latest, verify_artifacts,
)
from .config import load_config, resolve_path
from .data import load_data
from .evaluation import evaluate
from .preprocessing import preprocess
from .training import train_model

LOG = logging.getLogger("churn_pipeline")


def setup_logging(config, output):
    settings = config["logging"]
    for handler in LOG.handlers[:]:
        LOG.removeHandler(handler)
        handler.close()
    LOG.setLevel(settings["level"])
    LOG.propagate = False
    filename = settings["filename"]
    if Path(filename).name != filename:
        raise ValueError("logging.filename должен быть именем файла")
    file_handler = RotatingFileHandler(
        output / filename, maxBytes=settings["max_bytes"],
        backupCount=settings["backup_count"], encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in (file_handler, logging.StreamHandler()):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


def run_pipeline(config_path="config.json", mode=None, output=None, rebuild=False):
    config, root = load_config(config_path, mode, output)
    output = resolve_path(root, config["artifacts"]["output_dir"])
    started = time.perf_counter()
    with output_lock(output):
        setup_logging(config, output)
        try:
            LOG.info("Начало конвейера: mode=%s", config["mode"])
            frame, data_hash = load_data(config, root)
            splits = preprocess(frame, config)
            run_id, evidence = run_identity(config, data_hash, root)
            existing = output / "runs" / run_id
            if existing.exists() and not rebuild:
                metrics = verify_artifacts(existing)
                set_latest(output, run_id)
                LOG.info("CACHE_HIT: целостность проверена, обучение пропущено; run=%s", run_id)
                return {"run_id": run_id, "cached": True, "path": str(existing), "metrics": metrics}
            model = train_model(splits["train"], config)
            metrics, predictions = evaluate(model, splits, config)
            metrics["model_origin"] = "Trained in final_project"
            evidence["source"] = json.loads((root / "data/provenance.json").read_text(encoding="utf-8"))
            membership = pd.concat([
                part[[config["id_column"], config["target"]]].assign(split=name)
                for name, part in splits.items()
            ], ignore_index=True)
            directory = save_artifacts(output, run_id, model, metrics, config, predictions, membership, evidence)
            LOG.info("Test: %s; threshold=%.2f", metrics["test"], metrics["threshold"])
            LOG.info("Артефакты: %s; всего %.3f с", directory, time.perf_counter() - started)
            return {"run_id": run_id, "cached": False, "path": str(directory), "metrics": metrics}
        except Exception:
            LOG.exception("Ошибка конвейера; завершённые артефакты сохранены")
            raise
        finally:
            for handler in LOG.handlers[:]:
                LOG.removeHandler(handler)
                handler.close()
