import csv
import json
import os
import pathlib
import shutil
from timeit import default_timer as timer

import requests
import tensorflow as tf
from tensorflow import keras


URL = "http://192.168.0.206:5050"


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ParametersTracker(metaclass=Singleton):
    def __init__(self):
        self.tracking_precision = 0

    def get_model_parameters(self, no_steps):
        self.no_steps = no_steps
        self.steps_in_batch = int(self.no_steps * self.tracking_precision)
        self.batch_split = self.no_steps // self.steps_in_batch + (
            (self.no_steps % self.steps_in_batch) > 0
        )

    def get_first_val_step(self):
        return int(self.no_steps / self.batch_split)

    def write_parameters(self):
        return {
            "tracking_precision": self.tracking_precision,
            "no_steps": self.no_steps,
            "batch_split": self.batch_split,
            "max_batch_step": self.batch_split * (self.steps_in_batch - 1),
            "steps_in_batch": self.steps_in_batch,
        }


param_tracker = ParametersTracker()


class LiveLearningTracking(keras.callbacks.Callback):
    def __init__(self, tracking_precision):
        param_tracker.tracking_precision = tracking_precision
        self.step = 0

    def on_train_begin(self, logs=None):
        requests.delete(f"{URL}/clear")
        param_tracker.get_model_parameters(self.params["steps"])

        write_model_params(self.model, self.params)
        write_model_summary(self.model)

    def on_train_batch_end(self, batch, logs=None):
        if batch % param_tracker.batch_split == 0:
            write_data_train(
                self.step,
                batch,
                logs["accuracy"],
                logs["loss"],
            )
            self.step += 1

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.start = timer()

    def on_epoch_end(self, epoch, logs=None):
        self.stop = timer()
        if "val_loss" in logs.keys():
            write_data_val(
                self.step,
                logs["val_accuracy"],
                logs["val_loss"],
                epoch + 1,
                self.stop - self.start,
            )

    def on_train_end(self, logs=None):
        write_data_train(
            self.step,
            param_tracker.no_steps - param_tracker.batch_split,
            logs["accuracy"],
            logs["loss"],
        )
        if "val_loss" in logs.keys():
            write_data_val(
                self.step,
                logs["val_accuracy"],
                logs["val_loss"],
                self.epoch,
                0,
            )


def write_data_train(
    step,
    batch,
    train_loss,
    train_accuracy,
):
    requests.put(
        f"{URL}/train",
        json=(
            {
                "step": step,
                "batch": batch,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
            }
        ),
    )

    return (
        train_accuracy,
        train_loss,
    )


def write_data_val(
    step,
    val_loss,
    val_accuracy,
    epoch,
    epoch_time,
):
    requests.put(
        f"{URL}/val",
        json=(
            {
                "step": step,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
                "epoch": epoch,
                "epoch_time": epoch_time,
            }
        ),
    )

    return (
        val_accuracy,
        val_loss,
    )


def write_model_params(model, params):
    if params:
        params.update(param_tracker.write_parameters())
        params.update(
            {
                "no_tracked_steps": params["epochs"] * params["steps_in_batch"],
                "total_params": model.count_params(),
            }
        )

    requests.put(f"{URL}/params", json=params)

    return params


def write_model_summary(model):
    model_summary = str(model.to_json())
    layer_params = {
        "layers": [
            (layer.get_config(), {"no_params": layer.count_params()})
            for layer in model.layers
        ]
    }
    requests.put(f"{URL}/summary", json=json.loads(model_summary))
    requests.put(f"{URL}/layers", json=layer_params)

    return model.summary()
