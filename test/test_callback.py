import unittest

import tensorflow as tf
from callback import (
    ParametersTracker,
    write_data_train,
    write_data_val,
    write_model_params,
    write_model_summary,
)


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


class TestCallback(unittest.TestCase):
    def test_write_data_train(self):
        # GIVEN training data
        step = 1
        batch = 20
        train_loss = 0.85
        train_accuracy = 0.75

        # AND valid dictiorany output
        valid_dict = {
            "step": step,
            "batch": batch,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
        }

        # WHEN write_data_train() is ran with that input
        result = write_data_train(step, batch, train_loss, train_accuracy)

        # THEN function should return valid dictionary
        self.assertDictEqual(result, valid_dict)

    def test_write_data_val(self):
        # GIVEN validation data
        step = 10
        val_loss = 0.85
        val_accuracy = 0.75
        epoch = 2
        epoch_time = 4.25

        # AND valid dictiorany output
        valid_dict = {
            "step": step,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "epoch": epoch,
            "epoch_time": epoch_time,
        }

        # WHEN write_data_val() is ran with that input
        result = write_data_val(step, val_loss, val_accuracy, epoch, epoch_time)

        # THEN function should return valid dictionary
        self.assertDictEqual(result, valid_dict)

    def test_parameters_tracker(self):
        # GIVEN ParameterTracker class with tracking precision 0.01
        param_tracker = ParametersTracker(0.01)

        # WHEN get_model_parameters() is ran with known inputs
        inputs = {"steps": 1000, "epochs": 10}
        param_tracker.get_model_parameters(inputs)

        # THEN write_parameters() should return valid dictiorany
        result = param_tracker.write_parameters()

        valid_dict = {
            "tracking_precision": 0.01,
            "no_steps": 1000,
            "epochs": 10,
            "batch_split": 100,
            "max_batch_step": 900,
            "steps_in_batch": 10,
        }

        self.assertDictEqual(valid_dict, result)

    def test_write_model_params(self):
        # GIVEN model
        model = get_model()

        # AND ParametersTracker class with precision = 0.01 and known inputs
        param_tracker = ParametersTracker(0.01)
        inputs = {"steps": 1000, "epochs": 10}
        param_tracker.get_model_parameters(inputs)

        # WHEN write_model_params() is ran with that input
        result = write_model_params(model, param_tracker)

        # THEN it should return valid dictionary
        valid_dict = {
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

    def test_write_model_summary(self):
        # GIVEN model
        model = get_model()

        # WHEN write_model_summary() is ran with that input
        result = write_model_summary(model)

        # THEN is should return valid model summary
        valid_model_summary = str(model.to_json())

        self.assertEqual(result[0], valid_model_summary)

        # AND valid model_layers
        valid_model_layers = {"layers": [({"name": "flatten_1"})]}

        self.assertEqual(result[1].keys(), valid_model_layers.keys())


if __name__ == "__main__":
    unittest.main()
