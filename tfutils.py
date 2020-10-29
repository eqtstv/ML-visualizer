import tensorflow as tf
from tensorflow import keras
import pathlib
import csv
import os
import sys

LOGS_PATH = pathlib.Path(__file__).parent.joinpath("logs").resolve()


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


def add_eval(y, y_):
    """
    Add evaluation metrics.
    :param y: The predicted y, aka logits
    :param y_: The true labels
    :return: Add the accuracy and cross entropy to the tensorflow graph and return them
    """
    # Compute Accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Compute Cross Entropy
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

    return accuracy, cross_entropy


def write_data(
    accuracy,
    cross_entropy,
    feed_dict_train,
    feed_dict_val,
    step,
    step_range=5,
    filename="run_log.csv",
):
    """
    Writes accuracy and cross entropy value into the log file.
    :param accuracy:
    :param cross_entropy:
    :param feed_dict_train:
    :param feed_dict_val:
    :param step:
    :param step_range:
    :param filename: Name of the log file
    :return:
    """
    if step_range not in range(1, 1001):
        raise ValueError("Invalid step range. Please choose a value between 1 and 1000")

    # At the start, we delete the log residual log file from previous training
    if step == 0:
        if os.path.exists(filename):
            os.remove(filename)

    # Then we start logging inside the file
    elif step % step_range == 0:
        train_accuracy = accuracy.eval(feed_dict=feed_dict_train)
        val_accuracy = accuracy.eval(feed_dict=feed_dict_val)

        train_cross_entropy = cross_entropy.eval(feed_dict=feed_dict_train)
        val_cross_entropy = cross_entropy.eval(feed_dict=feed_dict_val)

        # Write CSV
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(
                [
                    step,
                    train_accuracy,
                    val_accuracy,
                    train_cross_entropy,
                    val_cross_entropy,
                ]
            )

        return train_accuracy, val_accuracy, train_cross_entropy, val_cross_entropy

    return None, None, None, None