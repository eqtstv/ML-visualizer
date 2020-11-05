import csv
import json
import os
import pathlib
import shutil
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow import keras

from app import config

LOGS_PATH = f"{pathlib.Path(__file__).parent.resolve()}/{config['logs_folder']}"
FILENAMES_DICT = config["filenames"]
TRACKING_PRECISION = config["tracking_precision"]


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ParametersTracker(metaclass=Singleton):
    def __init__(self, tracking_precision):
        self.tracking_precision = tracking_precision

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


param_tracker = ParametersTracker(TRACKING_PRECISION)


class LiveLearningTracking(keras.callbacks.Callback):
    def __init__(self):
        self.step = 0

    def on_train_begin(self, logs=None):
        clear_logs(LOGS_PATH, FILENAMES_DICT)
        param_tracker.get_model_parameters(self.params["steps"])

        write_model_params(self.params)
        write_model_summary(self.model)

    def on_train_batch_end(self, batch, logs=None):
        if batch % param_tracker.batch_split == 0:
            write_data_train(
                self.step,
                batch,
                logs["loss"],
                logs["accuracy"],
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
                logs["val_loss"],
                logs["val_accuracy"],
                epoch + 1,
                self.stop - self.start,
            )

    def on_train_end(self, logs=None):
        write_data_train(
            self.step,
            param_tracker.no_steps - param_tracker.batch_split,
            logs["loss"],
            logs["accuracy"],
        )
        write_data_val(
            self.step,
            logs["val_loss"],
            logs["val_accuracy"],
            self.epoch,
            0,
        )


def write_data_train(
    step,
    batch,
    train_loss,
    train_accuracy,
    filename=FILENAMES_DICT["log_train"],
):
    with open(f"{LOGS_PATH}/{filename}", "a", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(
            [
                step,
                batch,
                train_accuracy,
                train_loss,
            ]
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
    filename=FILENAMES_DICT["log_val"],
):
    with open(f"{LOGS_PATH}/{filename}", "a+", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(
            [
                step,
                val_accuracy,
                val_loss,
                epoch,
                epoch_time,
            ]
        )

    return (
        val_accuracy,
        val_loss,
    )


def write_model_summary(
    model,
    filename1=FILENAMES_DICT["model_summary"],
    filename2=FILENAMES_DICT["layers_summary"],
):
    model_summary = str(model.to_json())
    layer_params = {
        "layers": [
            (layer.get_config(), {"no_params": layer.count_params()})
            for layer in model.layers
        ]
    }

    with open(f"{LOGS_PATH}/{filename1}", "a+", newline="") as file:
        file.write(model_summary)

    with open(f"{LOGS_PATH}/{filename2}", "a+", newline="") as file:
        json.dump(layer_params, file, ensure_ascii=False, indent=4)

    return model.summary()


def write_model_params(params, filename=FILENAMES_DICT["model_params"]):
    if params:
        params.update(param_tracker.write_parameters())
        params.update({"no_tracked_steps": params["epochs"] * params["steps_in_batch"]})

    with open(f"{LOGS_PATH}/{filename}", "a+", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)

    return params


def clear_logs(folder, files):
    for key, filename in files.items():
        if os.path.exists(f"{folder}/{filename}"):
            os.remove(f"{folder}/{filename}")
