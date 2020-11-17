import csv
import json
import os
import pathlib
import shutil
import sys
from getpass import getpass, getuser
from timeit import default_timer as timer

import requests
import tensorflow as tf
from tensorflow import keras

# leave it for tests
# Change it when connecting models
from ml_visualizer.app import config

URL = f"http://{config['ip']}:{config['port']}"


def authenticate_user(email, password):
    user_data = {
        "email": email,
        "password": password,
    }
    return requests.post(f"{URL}/auth", json=user_data)


def write_data_train(
    step,
    batch,
    train_loss,
    train_accuracy,
):
    train_data = {
        "step": step,
        "batch": batch,
        "train_accuracy": train_accuracy,
        "train_loss": train_loss,
    }
    requests.put(
        f"{URL}/train",
        json=train_data,
        headers={"Authorization": f"Bearer {AuthToken.access_token}"},
    )

    return train_data


def write_data_val(
    step,
    val_loss,
    val_accuracy,
    epoch,
    epoch_time,
):
    val_data = {
        "step": step,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "epoch": epoch,
        "epoch_time": epoch_time,
    }

    requests.put(
        f"{URL}/val",
        json=val_data,
        headers={"Authorization": f"Bearer {AuthToken.access_token}"},
    )

    return val_data


def write_model_params(model, param_tracker):
    params = {}
    if param_tracker:
        params.update(param_tracker.write_parameters())
        params.update(
            {
                "no_tracked_steps": params["epochs"] * params["steps_in_batch"],
                "total_params": model.count_params(),
            }
        )

    requests.put(
        f"{URL}/params",
        json=params,
        headers={"Authorization": f"Bearer {AuthToken.access_token}"},
    )

    return params


def write_model_summary(model):
    model_summary = str(model.to_json())
    layer_params = {
        "layers": [
            (layer.get_config(), {"no_params": layer.count_params()})
            for layer in model.layers
        ]
    }
    requests.put(
        f"{URL}/summary",
        json=json.loads(model_summary),
        headers={"Authorization": f"Bearer {AuthToken.access_token}"},
    )
    requests.put(
        f"{URL}/layers",
        json=layer_params,
        headers={"Authorization": f"Bearer {AuthToken.access_token}"},
    )

    return model.to_json(), layer_params


def clear_training_data():
    return requests.delete(
        f"{URL}/clear", headers={"Authorization": f"Bearer {AuthToken.access_token}"}
    )


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AuthToken(metaclass=Singleton):
    def __init__(self, token):
        self.access_token = token


class ParametersTracker(metaclass=Singleton):
    def __init__(self, tracking_precision):
        self.tracking_precision = tracking_precision

    def get_model_parameters(self, params):
        self.no_steps = params["steps"]
        self.epochs = params["epochs"]
        self.steps_in_batch = int(self.no_steps * self.tracking_precision)
        self.batch_split = self.no_steps // self.steps_in_batch + (
            (self.no_steps % self.steps_in_batch) > 0
        )

    def write_parameters(self):
        return {
            "tracking_precision": self.tracking_precision,
            "no_steps": self.no_steps,
            "epochs": self.epochs,
            "batch_split": self.batch_split,
            "max_batch_step": self.batch_split * (self.steps_in_batch - 1),
            "steps_in_batch": self.steps_in_batch,
        }


class MLVisualizer(keras.callbacks.Callback):
    """
    Connect to MLVisualizer server

    Args:
        tracking_precision: Tracking precision of your learning process

    Returns:
        keras.callbacks.Callback
    """

    def __init__(self, tracking_precision=0.05):
        email = input("Email: ")
        password = getpass()

        response = authenticate_user(email, password)
        if "access_token" in response.json():
            AuthToken.access_token = response.json()["access_token"]
        else:
            print(response.json())
            sys.exit()

        project_name = input("Project name: ")

        self.param_tracker = ParametersTracker(tracking_precision)
        self.step = 0

    def on_train_begin(self, logs=None):
        clear_training_data()
        self.param_tracker.get_model_parameters(self.params)

        write_model_params(self.model, self.param_tracker)
        write_model_summary(self.model)

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.param_tracker.batch_split == 0:
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
            self.param_tracker.no_steps - self.param_tracker.batch_split,
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
