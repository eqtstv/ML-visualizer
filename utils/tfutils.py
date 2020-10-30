import tensorflow as tf
from tensorflow import keras
import pathlib
import csv
import os
import sys

LOGS_PATH = pathlib.Path(__file__).parent.parent.joinpath("logs").resolve()


class LossAndAccToCsvCallback(keras.callbacks.Callback):
    step = 0

    def on_train_batch_end(self, batch, logs=None):
        if batch % 28 == 0:
            write_data_train(
                self.step,
                logs["loss"],
                logs["accuracy"],
            )
            self.step += 1

    def on_epoch_end(self, epoch, logs=None):
        if "val_loss" in logs.keys():
            write_data_val(
                self.step - 1,
                logs["val_loss"],
                logs["val_accuracy"],
            )


def write_data_train(
    step,
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
    filename="run_log_val.csv",
):

    if step == 53:
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
            ]
        )

    return (
        val_accuracy,
        val_loss,
    )


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
