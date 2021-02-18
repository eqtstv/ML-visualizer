import json

import requests
from ml_visualizer.app import config


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Target(metaclass=Singleton):
    def __init__(self, url):
        self.url = url


def choose_target(target):
    if target == "local":
        Target.url = f"http://{config['ip']}:{config['port']}"
    if target == "cloud":
        Target.url = "https://live-ml-visualizer.herokuapp.com"


def authenticate_user(email, password):
    user_data = {
        "email": email,
        "password": password,
    }
    return requests.post(f"{Target.url}/auth", json=user_data)


def check_valid_project(
    auth_token, project_name, project_description="Project added from training"
):
    project = {
        "project_name": str(project_name),
        "project_description": str(project_description),
    }

    return requests.post(
        f"{Target.url}/project",
        json=project,
        headers={"Authorization": f"Bearer {auth_token}"},
    )


def create_new_project(auth_token, project_name, project_description):
    project = {
        "project_name": str(project_name),
        "project_description": str(project_description),
    }

    return requests.put(
        f"{Target.url}/project",
        json=project,
        headers={"Authorization": f"Bearer {auth_token}"},
    )


def write_data_train(
    auth_token,
    project_name,
    step,
    batch,
    train_loss,
    train_accuracy,
):
    train_data = {
        "project_name": project_name,
        "step": step,
        "batch": batch,
        "train_accuracy": train_accuracy,
        "train_loss": train_loss,
    }
    requests.put(
        f"{Target.url}/train",
        json=train_data,
        headers={"Authorization": f"Bearer {auth_token}"},
    )

    return train_data


def write_data_val(
    auth_token,
    project_name,
    step,
    val_loss,
    val_accuracy,
    epoch,
    epoch_time,
):
    val_data = {
        "project_name": project_name,
        "step": step,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "epoch": epoch,
        "epoch_time": epoch_time,
    }

    requests.put(
        f"{Target.url}/val",
        json=val_data,
        headers={"Authorization": f"Bearer {auth_token}"},
    )

    return val_data


def write_model_params(auth_token, model, param_tracker, project_name):
    params = {}
    if param_tracker:
        params.update(param_tracker.write_parameters())
        params.update(
            {
                "project_name": project_name,
                "no_tracked_steps": params["epochs"] * params["steps_in_batch"],
                "total_params": model.count_params(),
            }
        )

    requests.put(
        f"{Target.url}/params",
        json=params,
        headers={"Authorization": f"Bearer {auth_token}"},
    )

    return params


def write_model_summary(auth_token, model, project_name):
    model_summary = str(model.to_json())
    model_summary_json = json.loads(model_summary)
    model_summary_json.update({"project_name": project_name})

    layer_params = {
        "layers": [
            (layer.get_config(), {"no_params": layer.count_params()})
            for layer in model.layers
        ]
    }
    layer_params.update({"project_name": project_name})

    requests.put(
        f"{Target.url}/summary",
        json=model_summary_json,
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    requests.put(
        f"{Target.url}/layers",
        json=layer_params,
        headers={"Authorization": f"Bearer {auth_token}"},
    )

    return model_summary_json, layer_params


def clear_training_data(auth_token, project_name):
    return requests.delete(
        f"{Target.url}/clear",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"project_name": project_name},
    )
