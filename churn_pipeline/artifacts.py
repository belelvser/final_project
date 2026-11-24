"""Завершённые неизменяемые наборы артефактов и атомарный указатель latest."""
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from pathlib import Path

import joblib


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def environment():
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "pyarrow", "joblib"]
    return {"python": platform.python_version(), "platform": platform.system(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def run_identity(config, data_hash, root):
    semantic = {k: v for k, v in config.items() if k not in ("artifacts", "logging")}
    code_root = Path(__file__).resolve().parent.parent
    sources = sorted((code_root / "churn_pipeline").glob("*.py")) + [code_root / "main.py"]
    evidence = {
        "config": semantic, "input_sha256": data_hash, "environment": environment(),
        "code_sha256": {p.relative_to(code_root).as_posix(): sha256(p.read_bytes()) for p in sources},
        "data_provenance_sha256": sha256((Path(root) / "data/provenance.json").read_bytes()),
    }
    return sha256(json_bytes(evidence))[:24], evidence


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def output_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".pipeline.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Каталог занят: {path}. После аварии удалите lock только если процесс завершён.") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def verify_artifacts(directory):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected = {"model.joblib", "metrics.json", "config.json", "features.json",
                    "splits.csv", "predictions.csv", "provenance.json", "model.ubj", "imputer.json"}
        if set(manifest["files"]) != expected or manifest["run_id"] != directory.name:
            raise ValueError("Неверный состав manifest")
        for name, digest in manifest["files"].items():
            if sha256((directory / name).read_bytes()) != digest:
                raise ValueError(f"Нарушена целостность {name}")
        return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"Повреждён набор артефактов {directory}: {exc}. Используйте новый output-dir.") from exc


def set_latest(output, run_id):
    atomic_write(Path(output) / "latest.json", json_bytes({"run_id": run_id, "path": f"runs/{run_id}"}))


def save_artifacts(output, run_id, model, metrics, config, predictions, splits, evidence):
    output = Path(output)
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / run_id
    if destination.exists():
        old_metrics = verify_artifacts(destination)
        if old_metrics != metrics:
            raise ValueError("Повторное обучение дало другие метрики; сохранённый результат оставлен без изменений")
        if (destination / "predictions.csv").read_bytes() != predictions.to_csv(index=False).encode("utf-8"):
            raise ValueError("Повторное обучение дало другие предсказания; сохранённый результат не изменён")
        set_latest(output, run_id)
        return destination
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=runs) as temporary:
        staging = Path(temporary) / run_id
        staging.mkdir()
        joblib.dump(model, staging / "model.joblib", compress=3)
        # Стабильный формат деревьев дополняет Python-специфичный Pipeline.
        model.named_steps["clf"].save_model(staging / "model.ubj")
        (staging / "imputer.json").write_bytes(json_bytes({
            "strategy": config["preprocessing"]["imputation"],
            "statistics": model.named_steps["imputer"].statistics_.tolist(),
            "features": config["features"],
        }))
        items = {"metrics.json": metrics, "config.json": config,
                 "features.json": config["features"], "provenance.json": evidence}
        for name, value in items.items():
            (staging / name).write_bytes(json_bytes(value))
        predictions.to_csv(staging / "predictions.csv", index=False)
        splits.to_csv(staging / "splits.csv", index=False)
        manifest = {"run_id": run_id, "files": {
            p.name: sha256(p.read_bytes()) for p in sorted(staging.iterdir())
        }}
        (staging / "manifest.json").write_bytes(json_bytes(manifest))
        verify_artifacts(staging)
        # Публикуем только полный набор; предыдущие run-папки не перезаписываются.
        os.rename(staging, destination)
    set_latest(output, run_id)
    return destination
