import tensorflow as tf
from tensorflow import keras
import pathlib
import csv
import os
import sys
import json
from timeit import default_timer as timer

LOGS_PATH = pathlib.Path(__file__).parent.parent.joinpath("logs").resolve()

BATCH_SPLIT = 28

no_steps = 0


class LossAndAccToCsvCallback(keras.callbacks.Callback):
    def __init__(self):
        self.step = 0
        self.start = 0

    def on_train_begin(self, logs=None):
        global no_steps
        no_steps = self.params["steps"]
        get_model_params(self.params)
        get_model_summary(self.model)
        get_epoch_time(0, 0)

    def on_train_batch_end(self, batch, logs=None):
        if batch % BATCH_SPLIT == 0:
            write_data_train(
                self.step,
                batch,
                logs["loss"],
                logs["accuracy"],
            )
            self.step += 1

    def on_epoch_begin(self, epoch, logs=None):
        self.start = timer()

    def on_epoch_end(self, epoch, logs=None):
        self.stop = timer()
        if "val_loss" in logs.keys():
            write_data_val(
                self.step - 1,
                logs["val_loss"],
                logs["val_accuracy"],
                epoch,
                self.stop - self.start,
            )

    def on_train_end(self, logs=None):
        write_data_train(
            self.step,
            1484,
            logs["loss"],
            logs["accuracy"],
        )
        write_data_val(
            self.step - 1,
            logs["val_loss"],
            logs["val_accuracy"],
            9,  # epoch
            4.649970099999997,  # time
        )


def write_data_train(
    step,
    batch,
    train_loss,
    train_accuracy,
    filename="run_log_train.csv",
):

    if step == 0:
        if os.path.exists(f"{LOGS_PATH}/{filename}"):
            os.remove(f"{LOGS_PATH}/{filename}")

    # Write CSV
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
    filename="run_log_val.csv",
):
    if step == int(no_steps / BATCH_SPLIT):
        if os.path.exists(f"{LOGS_PATH}/{filename}"):
            os.remove(f"{LOGS_PATH}/{filename}")

    # Write CSV
    with open(f"{LOGS_PATH}/{filename}", "a", newline="") as file:
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


def get_model_summary(model, filename="model_summary.txt"):
    if model:
        if os.path.exists(f"{LOGS_PATH}/{filename}"):
            os.remove(f"{LOGS_PATH}/{filename}")

    with open(f"{LOGS_PATH}/{filename}", "a", newline="") as file:
        model.summary(print_fn=lambda x: file.write(x + "\n"))
    return model.summary()


def get_model_params(params, filename="model_params.json"):
    if params:
        if os.path.exists(f"{LOGS_PATH}/{filename}"):
            os.remove(f"{LOGS_PATH}/{filename}")

    with open(f"{LOGS_PATH}/{filename}", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    return params


def get_epoch_time(epoch, time, filename="epoch_times.csv"):
    if epoch == 0:
        if os.path.exists(f"{LOGS_PATH}/{filename}"):
            os.remove(f"{LOGS_PATH}/{filename}")

    # Write CSV
    with open(f"{LOGS_PATH}/{filename}", "a", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow([epoch, time])

    return (epoch, time)


def write_data_simple(
    step,
    train_loss,
    train_accuracy,
    test_loss,
    test_accuracy,
    filename="run_log.csv",
):

    if step == "1":
        if os.path.exists(f"{LOGS_PATH}/{filename}"):
            os.remove(f"{LOGS_PATH}/{filename}")

    # Write CSV
    with open(f"{LOGS_PATH}/{filename}", "a", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(
            [
                step,
                train_accuracy,
                train_loss,
                test_accuracy,
                test_loss,
            ]
        )

    return (
        train_accuracy,
        train_loss,
        test_accuracy,
        test_loss,
    )
