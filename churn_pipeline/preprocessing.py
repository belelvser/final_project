"""Разделение по абонентам и Pipeline с импутацией исключительно на train."""
import logging

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import validate_data

LOG = logging.getLogger(__name__)


def preprocess(frame, config):
    frame = validate_data(frame, config)
    target = config["target"]
    if config["split"]["strategy"] == "preserved":
        splits = {name: frame.loc[frame["split"] == name].copy()
                  for name in ("train", "validation", "test")}
    else:
        trainval, test = train_test_split(
            frame, test_size=config["split"]["test_size"],
            stratify=frame[target], random_state=config["seed"],
        )
        train, validation = train_test_split(
            trainval,
            test_size=config["split"]["validation_size"] / (1 - config["split"]["test_size"]),
            stratify=trainval[target], random_state=config["seed"],
        )
        splits = {"train": train, "validation": validation, "test": test}
    for name, part in splits.items():
        if part[target].nunique() != 2 or part[target].value_counts().min() < 2:
            raise ValueError(f"В {name} нужно не менее двух наблюдений каждого класса")
        LOG.info("%s: строк=%d, положительных=%d", name, len(part), part[target].sum())
    empty = splits["train"][config["features"]].isna().all()
    if empty.any():
        raise ValueError(f"Полностью пустые признаки в train: {empty.index[empty].tolist()}")
    return splits


def make_model(config, y_train):
    from xgboost import XGBClassifier

    params = dict(config["model"][config["mode"]])
    model = XGBClassifier(
        **params, objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=config["seed"], n_jobs=config["model"]["n_jobs"],
        scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy=config["preprocessing"]["imputation"])),
        ("clf", model),
    ])
