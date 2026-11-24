"""Обучение с уже выбранными в ДЗ №7 параметрами, без повторного Random Search."""
import logging
import time

from .preprocessing import make_model


def train_model(train, config):
    started = time.perf_counter()
    model = make_model(config, train[config["target"]])
    model.fit(train[config["features"]], train[config["target"]])
    logging.getLogger(__name__).info("Обучение завершено за %.3f с", time.perf_counter() - started)
    return model
