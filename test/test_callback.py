import json
import os
import unittest

import requests
import tensorflow as tf
from callback import (
    AuthToken,
    ParametersTracker,
    authenticate_user,
    check_valid_project,
    clear_training_data,
    create_new_project,
    write_data_train,
    write_data_val,
    write_model_params,
    write_model_summary,
)
from ml_visualizer.app import config
from ml_visualizer.database import Base, db_session

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def clear_data():
    meta = Base.metadata
    for table in reversed(meta.sorted_tables):
        db_session.execute(table.delete())
        db_session.commit()


def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


class TestAuth(unittest.TestCase):
    def test_proper_auth_token_init(self):
        token = AuthToken("mytoken")
        self.assertEqual(token.access_token, "mytoken")

    def test_auth_request_not_json(self):
        user = "testuser"
        r = requests.post(f"http://{config['ip']}:{config['port']}/auth", data=user)
        self.assertEqual(r.json(), {"msg": "Missing JSON in request."})

    def test_auth_missing_email_parameter(self):
        user = {"email": "", "password": "asd"}
        r = requests.post(f"http://{config['ip']}:{config['port']}/auth", json=user)
        self.assertEqual(r.json(), {"msg": "Missing email parameter."})

    def test_auth_missing_password_parameter(self):
        user = {"email": "asd@asd.pl", "password": ""}
        r = requests.post(f"http://{config['ip']}:{config['port']}/auth", json=user)
        self.assertEqual(r.json(), {"msg": "Missing password parameter."})

    def test_auth_wrong_email_or_password_parameter(self):
        user = {"email": "asd@asd.pl", "password": "asd"}
        r = requests.post(f"http://{config['ip']}:{config['port']}/auth", json=user)
        self.assertEqual(r.json(), {"msg": "Bad email or password."})

    def test_user_authenticate_user(self):
        user = {"email": "asd@asd.pl", "name": "asd", "password": "asd"}
        r = requests.put(f"http://{config['ip']}:{config['port']}/signup", json=user)

        AuthToken.access_token = authenticate_user("asd@asd.pl", "asd").json()[
            "access_token"
        ]

        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(AuthToken.access_token), 283)
        clear_data()


class TestProjects(unittest.TestCase):
    def test_check_valid_project_invalid_name(self):
        project_name = "no_project"
        project_description = "no project description"

        r = check_valid_project(project_name, project_description)

        self.assertEqual(
            r.json(),
            {
                "msg": "Invalid project name!\n \
                Do you want to create new project? (y/n)"
            },
        )

    def test_check_create_new_project_valid(self):
        project_name = "myproject"
        project_description = "myproject description"

        r = create_new_project(project_name, project_description)
        self.assertEqual(r.status_code, 200)
        clear_data()


class TestClearData(unittest.TestCase):
    def test_clear_data(self):
        user = {"email": "asd@asd.pl", "name": "asd", "password": "asd"}
        r = requests.put(f"http://{config['ip']}:{config['port']}/signup", json=user)

        AuthToken.access_token = authenticate_user("asd@asd.pl", "asd").json()[
            "access_token"
        ]
        r = clear_training_data("my_project")
        self.assertEqual(r.status_code, 200)
        clear_data()


class TestCallback(unittest.TestCase):
    def test_write_data_train(self):
        # GIVEN training data
        project = "my_project"
        step = 1
        batch = 20
        train_loss = 0.85
        train_accuracy = 0.75

        # AND valid dictiorany output
        valid_dict = {
            "project_name": project,
            "step": step,
            "batch": batch,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
        }
        user = {"email": "asd@asd.pl", "name": "asd", "password": "asd"}
        requests.put(f"http://{config['ip']}:{config['port']}/signup", json=user)

        AuthToken.access_token = authenticate_user("asd@asd.pl", "asd").json()[
            "access_token"
        ]

        # WHEN write_data_train() is ran with that input
        result = write_data_train(project, step, batch, train_loss, train_accuracy)

        # THEN function should return valid dictionary
        self.assertDictEqual(result, valid_dict)
        clear_data()

    def test_write_data_val(self):
        # GIVEN validation data
        project = "my_project"
        step = 10
        val_loss = 0.85
        val_accuracy = 0.75
        epoch = 2
        epoch_time = 4.25

        # AND valid dictiorany output
        valid_dict = {
            "project_name": project,
            "step": step,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "epoch": epoch,
            "epoch_time": epoch_time,
        }
        user = {"email": "asd@asd.pl", "name": "asd", "password": "asd"}
        requests.put(f"http://{config['ip']}:{config['port']}/signup", json=user)

        AuthToken.access_token = authenticate_user("asd@asd.pl", "asd").json()[
            "access_token"
        ]
        # WHEN write_data_val() is ran with that input
        result = write_data_val(
            project, step, val_loss, val_accuracy, epoch, epoch_time
        )

        # THEN function should return valid dictionary
        self.assertDictEqual(result, valid_dict)
        clear_data()

    def test_parameters_tracker(self):
        # GIVEN ParameterTracker class with tracking precision 0.01
        param_tracker = ParametersTracker(0.01)

        # WHEN get_model_parameters() is ran with known inputs
        inputs = {"steps": 1000, "epochs": 10}
        param_tracker.get_model_parameters(inputs)

        # THEN write_parameters() should return valid dictiorany
        result = param_tracker.write_parameters()
        user = {"email": "asd@asd.pl", "name": "asd", "password": "asd"}
        requests.put(f"http://{config['ip']}:{config['port']}/signup", json=user)

        AuthToken.access_token = authenticate_user("asd@asd.pl", "asd").json()[
            "access_token"
        ]
        valid_dict = {
            "tracking_precision": 0.01,
            "no_steps": 1000,
            "epochs": 10,
            "batch_split": 100,
            "max_batch_step": 900,
            "steps_in_batch": 10,
        }

        self.assertDictEqual(valid_dict, result)
        clear_data()

    def test_write_model_params(self):
        # GIVEN model
        model = get_model()
        project = "my_project"

        # AND ParametersTracker class with precision = 0.01 and known inputs
        param_tracker = ParametersTracker(0.01)
        inputs = {"steps": 1000, "epochs": 10}
        param_tracker.get_model_parameters(inputs)

        # WHEN write_model_params() is ran with that input
        result = write_model_params(model, param_tracker, project)

        # THEN it should return valid dictionary
        valid_dict = {
            "project_name": project,
            "tracking_precision": 0.01,
            "no_steps": 1000,
            "epochs": 10,
            "batch_split": 100,
            "max_batch_step": 900,
            "steps_in_batch": 10,
            "no_tracked_steps": 100,
            "total_params": 101770,
        }

        self.assertDictEqual(result, valid_dict)
        clear_data()

    def test_write_model_summary(self):
        # GIVEN model
        model = get_model()
        project_name = "my_project"

        # WHEN write_model_summary() is ran with that input
        result = write_model_summary(model, project_name)

        # THEN is should return valid model summary
        model_summary = str(model.to_json())
        valid_model_summary = json.loads(model_summary)
        valid_model_summary.update({"project_name": project_name})

        self.assertEqual(result[0], valid_model_summary)

        # AND valid model_layers
        valid_model_layers = {
            "layers": [({"name": "flatten_1"})],
            "project_name": project_name,
        }

        self.assertEqual(result[1].keys(), valid_model_layers.keys())
        clear_data()


if __name__ == "__main__":
    unittest.main()
