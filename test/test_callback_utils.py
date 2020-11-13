import json
import unittest

import dash_html_components as html
import pandas as pd
import plotly
from ml_visualizer.callbacks_utils import (
    get_input_layer_info,
    get_layers,
    update_current_value,
    update_graph,
)

run_log_json_no_val = {
    "columns": [
        "step",
        "batch",
        "train_accuracy",
        "train_loss",
        "val_accuracy",
        "val_loss",
        "epoch",
        "epoch_time",
    ],
    "index": [0, 1],
    "data": [
        [0, 0, 0.1875, 2.5382819176, None, None, None, None],
        [1, 10, 0.2, 2.2, None, None, None, None],
    ],
}
run_log_json_with_val = {
    "columns": [
        "step",
        "batch",
        "train_accuracy",
        "train_loss",
        "val_accuracy",
        "val_loss",
        "epoch",
        "epoch_time",
    ],
    "index": [0, 1, 2],
    "data": [
        [0, 0, 0.1875, 2.5382819176, None, None, None, None],
        [1, 10, 0.2, 2.2, None, None, None, None],
        [2, 750, 0.90625, 0.2020742446, 0.836499989, 0.4435300827, 1.0, 19.0966033],
    ],
}

model_summary = {
    "class_name": "Sequential",
    "config": {
        "name": "sequential",
        "layers": [
            {
                "class_name": "InputLayer",
                "config": {
                    "batch_input_shape": [None, 28, 28],
                    "dtype": "float32",
                    "sparse": False,
                    "ragged": False,
                    "name": "flatten_input",
                },
            },
            {
                "class_name": "Flatten",
                "config": {
                    "name": "flatten",
                    "trainable": True,
                    "batch_input_shape": [None, 28, 28],
                    "dtype": "float32",
                    "data_format": "channels_last",
                },
            },
            {
                "class_name": "Dense",
                "config": {
                    "name": "dense",
                    "trainable": True,
                    "dtype": "float32",
                    "units": 128,
                    "activation": "relu",
                    "use_bias": True,
                    "kernel_initializer": {
                        "class_name": "GlorotUniform",
                        "config": {"seed": None},
                    },
                    "bias_initializer": {"class_name": "Zeros", "config": {}},
                    "kernel_regularizer": None,
                    "bias_regularizer": None,
                    "activity_regularizer": None,
                    "kernel_constraint": None,
                    "bias_constraint": None,
                },
            },
            {
                "class_name": "Dense",
                "config": {
                    "name": "dense_1",
                    "trainable": True,
                    "dtype": "float32",
                    "units": 10,
                    "activation": "linear",
                    "use_bias": True,
                    "kernel_initializer": {
                        "class_name": "GlorotUniform",
                        "config": {"seed": None},
                    },
                    "bias_initializer": {"class_name": "Zeros", "config": {}},
                    "kernel_regularizer": None,
                    "bias_regularizer": None,
                    "activity_regularizer": None,
                    "kernel_constraint": None,
                    "bias_constraint": None,
                },
            },
        ],
    },
    "keras_version": "2.4.0",
    "backend": "tensorflow",
}


class TestUpdateGraph(unittest.TestCase):
    def test_graph_names(self):
        # GIVEN known graph parameters
        graph_params = [
            "example-graph",
            "Example Graph",
            "train_accuracy",
            "train_val",
            json.dumps(run_log_json_no_val),
            "Example_axis",
        ]

        # WHEN graph is initiated with that parameters
        graph = update_graph(*graph_params)

        # THEN grph should have proper name values
        self.assertEqual(graph.id, "example-graph")
        self.assertEqual(graph.figure.layout.title.text, "Example Graph")

    def test_graph_json(self):
        # GIVEN known graph parameters
        graph_params = [
            "example-graph",
            "Example Graph",
            "train_accuracy",
            "train_val",
            json.dumps(run_log_json_no_val),
            "Example_axis",
        ]

        # WHEN graph is initiated with that parameters
        graph = update_graph(*graph_params)

        self.assertIsInstance(graph.figure.data[0], plotly.graph_objs._scatter.Scatter)
        self.assertIsInstance(graph.figure.data[1], plotly.graph_objs._scatter.Scatter)


class TestUpdateCurrentValueAccuracy(unittest.TestCase):
    def test_update_values_no_validation(self):
        # GIVEN known value parameters
        params = [
            "train_accuracy",
            "val_accuracy",
            "Accuracy",
            json.dumps(run_log_json_no_val),
        ]

        # WHEN update_current_value is ran with that input
        result = update_current_value(*params)

        # THEN it should return proper name and value
        self.assertEqual(result[0].children, "Current Accuracy:")
        self.assertEqual(result[1].children, "Training: 0.2000")

    def test_update_value_with_validation(self):
        # GIVEN known value parameters
        params = [
            "train_accuracy",
            "val_accuracy",
            "Accuracy",
            json.dumps(run_log_json_with_val),
        ]

        # WHEN update_current_value is ran with that input
        result = update_current_value(*params)

        # THEN it should return proper name and value
        self.assertEqual(result[0].children, "Current Accuracy:")
        self.assertEqual(result[1].children, "Training: 0.9063")
        self.assertEqual(result[2].children, "Validation: 0.8365")


class TestUpdateCurrentValueLoss(unittest.TestCase):
    def test_update_values_no_validation(self):
        # GIVEN known value parameters
        params = [
            "train_loss",
            "val_loss",
            "Loss",
            json.dumps(run_log_json_no_val),
        ]

        # WHEN update_current_value is ran with that input
        result = update_current_value(*params)

        # THEN it should return proper name and value
        self.assertEqual(result[0].children, "Current Loss:")
        self.assertEqual(result[1].children, "Training: 2.2000")

    def test_update_value_with_validation(self):
        # GIVEN known value parameters
        params = [
            "train_loss",
            "val_loss",
            "Loss",
            json.dumps(run_log_json_with_val),
        ]

        # WHEN update_current_value is ran with that input
        result = update_current_value(*params)

        # THEN it should return proper name and value
        self.assertEqual(result[0].children, "Current Loss:")
        self.assertEqual(result[1].children, "Training: 0.2021")
        self.assertEqual(result[2].children, "Validation: 0.4435")


class TestGetInputLayerInfo(unittest.TestCase):
    def test_get_input_layer_info(self):
        # GIVEN model known model_summary
        # WHEN get_input_layer_info() is ran with that input
        result = get_input_layer_info(model_summary)

        # THEN it should return valid result
        valid_result = {
            "class_name": "InputLayer",
            "name": "flatten_input",
            "input_shape": [None, 28, 28],
        }
        self.assertEqual(result, valid_result)


class TestGetLayers(unittest.TestCase):
    def test_get_layers(self):
        # GIVEN model known model_summary
        # WHEN get_layers() is ran with that input
        result = get_layers(model_summary)

        # THEN it should return valid result
        valid_result = [
            {"Type": "InputLayer", "name": "flatten_input"},
            {"Type": "Flatten", "name": "flatten"},
            {"Type": "Dense", "name": "dense", "units": 128, "activation": "relu"},
            {"Type": "Dense", "name": "dense_1", "units": 10, "activation": "linear"},
        ]
        self.assertEqual(result, valid_result)