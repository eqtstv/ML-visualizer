import abc
import sys
from getpass import getpass
from timeit import default_timer as timer

from tensorflow import keras

from mlvisualizer.utils import (
    authenticate_user,
    check_valid_project,
    clear_training_data,
    create_new_project,
    write_data_train,
    write_data_val,
    write_model_params,
    write_model_summary,
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
    def __init__(self, tracking_precision=0.01):
        email = input("Email: ")
        password = getpass()

        auth_response = authenticate_user(email, password)

        if "access_token" in auth_response.json():
            AuthToken.access_token = auth_response.json()["access_token"]
            print("\nAuthentication successful.\n")
        else:
            print(f"\n{auth_response.json()['msg']}\n")
            sys.exit()
        check_project_name = str(input("Project name: "))
        project_response = check_valid_project(
            AuthToken.access_token, check_project_name
        )

        if project_response.status_code == 200:
            print("\nProject selected\n")
            self.project_name = check_project_name
        else:
            print(f"\n{project_response.json()['msg']}\n")
            decide = input()

            if decide == "y":
                new_name = input("Project name: ")
                new_description = input("Project description: ")
                project_response = create_new_project(
                    AuthToken.access_token, new_name, new_description
                )

            if project_response.status_code == 200:
                print("\nProject successfully created.\n")
                self.project_name = new_name
                print("Start training? y/n")
                decide = input()

                if decide != "y":
                    sys.exit()

            else:
                print(f"\n{project_response.json()['msg']}\n")
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
