import abc
import os
import sys
from getpass import getpass
from timeit import default_timer as timer

from tensorflow import keras

from mlvisualizer.utils import (
    authenticate_user,
    check_valid_project,
    choose_target,
    clear_training_data,
    create_new_project,
    write_data_train,
    write_data_val,
    write_model_params,
    write_model_summary,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
    def __init__(self, tracking_precision=0.01):
        target = input("Choose target (local/cloud): ")
        choose_target(target)

        email = input("Email: ")
        password = getpass()
        AuthToken.access_token = authenticate_user(email, password)

        check_project_name = str(input("Project name: "))

        if check_valid_project(AuthToken.access_token, check_project_name):
            self.project_name = check_project_name
        else:
            print("Do you want to create new project? (y/n)")
            decide = input()

            if decide == "y":
                self.project_name = create_new_project(AuthToken.access_token)
            else:
                sys.exit()

        print("Start training? y/n")
        decide = input()

        if decide != "y":
            sys.exit()

        self.param_tracker = ParametersTracker(tracking_precision)
        self.step = 0

    @abc.abstractmethod
    def on_train_begin(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_train_batch_end(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_epoch_begin(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_epoch_end(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_train_end(self):
        raise NotImplementedError


class BatchTracker(MLVisualizer):
    def on_train_begin(self, logs=None):
        clear_training_data(AuthToken.access_token, self.project_name)
        self.param_tracker.get_model_parameters(self.params)

        write_model_params(
            AuthToken.access_token, self.model, self.param_tracker, self.project_name
        )
        write_model_summary(AuthToken.access_token, self.model, self.project_name)

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.param_tracker.batch_split == 0:
            write_data_train(
                AuthToken.access_token,
                self.project_name,
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
                AuthToken.access_token,
                self.project_name,
                self.step,
                logs["val_accuracy"],
                logs["val_loss"],
                epoch + 1,
                self.stop - self.start,
            )

    def on_train_end(self, logs=None):
        write_data_train(
            AuthToken.access_token,
            self.project_name,
            self.step,
            self.param_tracker.batch_split * (self.param_tracker.steps_in_batch - 1),
            logs["accuracy"],
            logs["loss"],
        )
        if "val_loss" in logs.keys():
            write_data_val(
                AuthToken.access_token,
                self.project_name,
                self.step,
                logs["val_accuracy"],
                logs["val_loss"],
                self.epoch,
                0,
            )


class EpochTracker(MLVisualizer):
    @abc.abstractmethod
    def on_train_begin(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_train_batch_end(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_epoch_begin(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_epoch_end(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_train_end(self):
        raise NotImplementedError
