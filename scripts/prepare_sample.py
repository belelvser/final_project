"""Воспроизвести выборку из исходной витрины; по умолчанию только проверка."""
import argparse
import io
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from churn_pipeline.artifacts import atomic_write, json_bytes, sha256  # noqa: E402
from churn_pipeline.config import load_config, resolve_path  # noqa: E402
from churn_pipeline.data import load_data, validate_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Пересоздать sample и обновить его SHA256")
    args = parser.parse_args()
    config, root = load_config(ROOT / "config.json", mode="full")
    frame, _ = load_data(config, root)
    frame = validate_data(frame, config)
    pieces = []
    for name, size in (("train", 3600), ("validation", 1200), ("test", 1200)):
        part = frame.loc[frame["split"] == name].sort_values(config["id_column"])
        sample, _ = train_test_split(part, train_size=size, stratify=part[config["target"]],
                                     random_state=config["seed"])
        pieces.append(sample)
    sample = pd.concat(pieces).sort_values(config["id_column"]).reset_index(drop=True)
    # Сохраняем тот же контракт и порядок столбцов, что в архивной витрине.
    order = [config["id_column"], *config["features"], config["target"], "split"]
    sample = sample[order]
    target = resolve_path(root, config["data"]["demo"]["path"])
    if not args.write:
        actual = pd.read_parquet(target)
        pd.testing.assert_frame_equal(sample, actual, check_dtype=False)
        if sha256(target.read_bytes()) != config["data"]["demo"]["sha256"]:
            raise ValueError("Неверный SHA256 sample")
        print("OK: все 6000 строк, признаки, метки и split воспроизводятся из полной витрины")
        return
    buffer = io.BytesIO()
    sample.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    original = json.loads((root / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "data/provenance.json").read_text(encoding="utf-8"))
    original["data"]["demo"]["sha256"] = sha256(payload)
    provenance["sample_sha256"] = sha256(payload)
    atomic_write(target, payload)
    atomic_write(root / "config.json", json_bytes(original))
    atomic_write(root / "data/provenance.json", json_bytes(provenance))
    print("Сохранено:", target)


if __name__ == "__main__":
    main()
