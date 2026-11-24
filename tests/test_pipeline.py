"""Тесты на реальной учебной выборке из пользовательского архива."""
import copy
import json
import shutil
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from churn_pipeline.artifacts import output_lock, save_artifacts, sha256, verify_artifacts
from churn_pipeline.config import FEATURES, load_config
from churn_pipeline.data import load_data, validate_data
from churn_pipeline.evaluation import choose_threshold, evaluate
from churn_pipeline.pipeline import run_pipeline
from churn_pipeline.preprocessing import preprocess
from churn_pipeline.training import train_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "config.json")[0]


@pytest.fixture
def frame(config):
    return load_data(config, ROOT)[0]


def test_provenance_and_disjoint_subscribers(config, frame):
    provenance = json.loads((ROOT / "data/provenance.json").read_text(encoding="utf-8"))
    assert provenance["sample_sha256"] == config["data"]["demo"]["sha256"]
    assert provenance["feature_window"][1] == provenance["label_window"][0]
    assert config["features"] == FEATURES
    assert len(frame) == 6000
    parts = preprocess(frame, config)
    assert [len(parts[x]) for x in ("train", "validation", "test")] == [3600, 1200, 1200]
    ids = [set(p.subscriber_key) for p in parts.values()]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]


def test_full_dataset_is_self_contained(tmp_path):
    # Имитируем получение проекта без соседнего архива и домашних работ.
    (tmp_path / "data").mkdir()
    shutil.copyfile(ROOT / "config.json", tmp_path / "config.json")
    shutil.copyfile(ROOT / "data/full.parquet", tmp_path / "data/full.parquet")
    config, root = load_config(tmp_path / "config.json", mode="full")
    frame, digest = load_data(config, root)
    provenance = json.loads((ROOT / "data/provenance.json").read_text(encoding="utf-8"))
    assert digest == provenance["full_sha256"] == provenance["source_member_sha256"]
    assert len(frame) == provenance["source_rows"] == 186463
    assert frame["churn_proxy"].sum() == 16449
    parts = preprocess(frame, config)
    assert [len(parts[x]) for x in ("train", "validation", "test")] == [111877, 37293, 37293]
    assert frame["subscriber_key"].is_unique


@pytest.mark.parametrize("defect", ["missing", "duplicate", "empty_id", "bad_target", "one_class",
                                     "infinity", "text", "negative", "bad_split", "null_split", "inactive"])
def test_rejects_invalid_data(config, frame, defect):
    data = frame.copy()
    if defect == "missing":
        data = data.drop(columns=[FEATURES[0]])
    elif defect == "duplicate":
        data.loc[1, "subscriber_key"] = data.loc[0, "subscriber_key"]
    elif defect == "empty_id":
        data.loc[0, "subscriber_key"] = " "
    elif defect == "bad_target":
        data.loc[0, "churn_proxy"] = 2
    elif defect == "one_class":
        data["churn_proxy"] = 0
    elif defect == "infinity":
        data.loc[0, FEATURES[1]] = np.inf
    elif defect == "text":
        data[FEATURES[1]] = data[FEATURES[1]].astype(object)
        data.loc[0, FEATURES[1]] = "broken"
    elif defect == "negative":
        data.loc[0, FEATURES[1]] = -1
    elif defect == "bad_split":
        data.loc[0, "split"] = "future"
    elif defect == "null_split":
        data.loc[0, "split"] = None
    elif defect == "inactive":
        data.loc[0, FEATURES[0]] = 0
    with pytest.raises(ValueError):
        validate_data(data, config)


@pytest.mark.parametrize("feature", ["retention_contact", "churn_proxy", "subscriber_key", "churn_day_index"])
def test_config_rejects_leakage(config, tmp_path, feature):
    config["features"].append(feature)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="признаки"):
        load_config(path)


def test_source_hash_and_missing_file(config):
    config["data"]["demo"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        load_data(config, ROOT)
    config["data"]["demo"]["path"] = "data/not_present.parquet"
    with pytest.raises(FileNotFoundError):
        load_data(config, ROOT)


def test_stratified_split_is_reproducible(config, frame):
    config["split"] = {"strategy": "stratified", "test_size": 0.15, "validation_size": 0.25}
    first = preprocess(frame, config)
    second = preprocess(frame, config)
    assert [len(first[x]) for x in ("train", "validation", "test")] == [3600, 1500, 900]
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_imputer_fits_only_train_and_threshold_ignores_test(config, frame):
    parts = preprocess(frame, config)
    model = train_model(parts["train"], config)
    np.testing.assert_allclose(model.named_steps["imputer"].statistics_,
                               parts["train"][FEATURES].median().to_numpy())
    before, _ = evaluate(model, parts, config)
    changed = copy.deepcopy(parts)
    changed["test"]["churn_proxy"] = 1 - changed["test"]["churn_proxy"]
    changed["test"][FEATURES] = changed["test"][FEATURES] * 100
    after, _ = evaluate(model, changed, config)
    assert before["threshold"] == after["threshold"]
    assert before["validation"] == after["validation"]
    assert before["test"] != after["test"]
    assert choose_threshold([0, 1, 1], np.array([0.1, 0.6, 0.8]), [0.3, 0.5, 0.7]) == 0.3


def test_empty_training_feature_fails(config, frame):
    frame.loc[frame["split"] == "train", FEATURES[-1]] = np.nan
    with pytest.raises(ValueError, match="Полностью пустые"):
        preprocess(frame, config)


def test_end_to_end_cache_rebuild_and_inference(tmp_path, config, frame, monkeypatch):
    output = tmp_path / "artifacts"
    first = run_pipeline(ROOT / "config.json", output=output)
    directory = Path(first["path"])
    hashes = {p.name: sha256(p.read_bytes()) for p in directory.iterdir()}
    second = run_pipeline(ROOT / "config.json", output=output, rebuild=True)
    assert second["run_id"] == first["run_id"]
    assert second["metrics"] == first["metrics"]
    assert hashes == {p.name: sha256(p.read_bytes()) for p in directory.iterdir()}
    assert len(list((output / "runs").iterdir())) == 1
    loaded = joblib.load(directory / "model.joblib")
    parts = preprocess(frame, config)
    actual = loaded.predict_proba(parts["test"][FEATURES])[:, 1]
    expected = pd.read_csv(directory / "predictions.csv").probability.to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    from xgboost import XGBClassifier
    portable = XGBClassifier()
    portable.load_model(directory / "model.ubj")
    state = json.loads((directory / "imputer.json").read_text())
    matrix = parts["test"][state["features"]].fillna(dict(zip(state["features"], state["statistics"])))
    np.testing.assert_allclose(portable.predict_proba(matrix.to_numpy())[:, 1], actual, atol=1e-7)
    monkeypatch.setattr("churn_pipeline.pipeline.train_model", lambda *args: pytest.fail("Кэш запустил обучение"))
    third = run_pipeline(ROOT / "config.json", output=output)
    assert third["cached"] is True
    assert hashes == {p.name: sha256(p.read_bytes()) for p in directory.iterdir()}
    (directory / "model.joblib").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Повреждён"):
        run_pipeline(ROOT / "config.json", output=output)


def test_failed_save_keeps_previous_result(tmp_path, monkeypatch, config, frame):
    output = tmp_path / "output"
    predictions = pd.DataFrame({"probability": [0.1]})
    splits = pd.DataFrame({"split": ["test"]})
    model = train_model(preprocess(frame, config)["train"], config)
    previous = save_artifacts(output, "old", model, {"score": 1}, config, predictions, splits, {})
    pointer = (output / "latest.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("churn_pipeline.artifacts.joblib.dump", fail)
    with pytest.raises(OSError, match="simulated"):
        save_artifacts(output, "new", {}, {}, config, predictions, splits, {})
    assert verify_artifacts(previous) == {"score": 1}
    assert (output / "latest.json").read_bytes() == pointer
    assert not (output / "runs/new").exists()
    assert not list((output / "runs").glob(".staging-*"))


def test_concurrent_output_is_locked(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="занят"):
            with output_lock(tmp_path):
                pass
    assert not (tmp_path / ".pipeline.lock").exists()


def test_cli_missing_config_returns_nonzero(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--config", str(tmp_path / "absent.json")],
                            capture_output=True, encoding="utf-8", errors="replace")
    assert result.returncode == 1
    assert result.stderr
