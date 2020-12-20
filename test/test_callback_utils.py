import json
import unittest

import dash_html_components as html
import plotly
from ml_visualizer.dash_app.callbacks_utils import (
    get_input_layer_info,
    get_layers,
    get_model_summary_divs,
    update_current_value,
    update_graph,
    update_interval_log,
    update_progress_bars,
    update_progress_display,
)

run_log_json_no_validation = {
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
run_log_json_with_validation = {
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
    "class_name": {"0": "Sequential"},
    "config": {
        "0": '{"name": "sequential", "layers": \
        [{"class_name": "InputLayer", \
        "config": {"batch_input_shape": [null, 28, 28],\
        "dtype": "float32", "sparse": false, \
        "ragged": false, "name": "flatten_input"}},\
        {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true,\
        "batch_input_shape": [null, 28, 28], "dtype": "float32", \
        "data_format": "channels_last"}},\
        {"class_name": "Dense", "config": \
        {"name": "dense", "trainable": true, "dtype": "float32", \
        "units": 128, "activation": "relu",\
        "use_bias": true, "kernel_initializer": \
        {"class_name": "GlorotUniform", "config": {"seed": null}},\
        "bias_initializer": {"class_name": "Zeros", "config": {}}, \
        "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer":\
        null, "kernel_constraint": null, "bias_constraint": null}}, \
        {"class_name": "Dense", "config": {"name": "dense_1", \
        "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", \
        "use_bias": true, "kernel_initializer": \
        {"class_name": "GlorotUniform", "config": {"seed": null}},\
        "bias_initializer": {"class_name": "Zeros", "config": {}},\
        "kernel_regularizer": null, "bias_regularizer": null,\
        "activity_regularizer": null, "kernel_constraint":\
        null, "bias_constraint": null}}]}'
    },
}

model_params = {
    "tracking_precision": {"0": 0.01},
    "no_steps": {"0": 1000},
    "epochs": {"0": 10},
    "batch_split": {"0": 100},
    "max_batch_step": {"0": 900},
    "steps_in_batch": {"0": 10},
    "no_tracked_steps": {"0": 100},
    "total_params": {"0": 101770},
}


class TestUpdateGraph(unittest.TestCase):
    def test_graph_names(self):
        # GIVEN known graph parameters
        graph_params = [
            "example-graph",
            "Example Graph",
            "train_accuracy",
            "train_val",
            json.dumps(run_log_json_no_validation),
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
            json.dumps(run_log_json_no_validation),
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
            json.dumps(run_log_json_no_validation),
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
            json.dumps(run_log_json_with_validation),
        ]

        # WHEN update_current_value is ran with that input
        result = update_current_value(*params)

        # THEN it should return proper name and value
        self.assertEqual(result[0].children, "Current Accuracy:")
        self.assertEqual(result[1].children, "Training: 0.9063")
        self.assertEqual(" ".join(result[2].children.split()), "Validation: 0.8365")


class TestUpdateCurrentValueLoss(unittest.TestCase):
    def test_update_values_no_validation(self):
        # GIVEN known value parameters
        params = [
            "train_loss",
            "val_loss",
            "Loss",
            json.dumps(run_log_json_no_validation),
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
            json.dumps(run_log_json_with_validation),
        ]

        # WHEN update_current_value is ran with that input
        result = update_current_value(*params)

        # THEN it should return proper name and value
        self.assertEqual(result[0].children, "Current Loss:")
        self.assertEqual(result[1].children, "Training: 0.2021")
        self.assertEqual(" ".join(result[2].children.split()), "Validation: 0.4435")


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


class TestIntervalLog(unittest.TestCase):
    def test_update_interval_log(self):
        # GIVEN known inputs
        inputs = ["fast", "regular", "slow", "no"]
        results = [500, 1000, 5000, 86400000]

        # WHEN update_interval_log() is ran with that input
        result1 = update_interval_log(inputs[0])
        result2 = update_interval_log(inputs[1])
        result3 = update_interval_log(inputs[2])
        result4 = update_interval_log(inputs[3])

        # THEN it should return valid results
        self.assertEqual(result1, results[0])
        self.assertEqual(result2, results[1])
        self.assertEqual(result3, results[2])
        self.assertEqual(result4, results[3])


class TestProgressBars(unittest.TestCase):
    def test_update_progress_bars_no_validation(self):
        # GIVEN known inputs
        # WHEN update_progress_bars() is ran with that input
        result = update_progress_bars(
            json.dumps(run_log_json_no_validation), model_params
        )

        # THEN it should return valid results
        valid_result = (1.1111111111111112, "", 1.0, "")

        self.assertEqual(result, valid_result)

    def test_update_progress_bars_with_validation(self):
        # GIVEN known inputs
        # WHEN update_progress_bars() is ran with that input
        result = update_progress_bars(
            json.dumps(run_log_json_with_validation), model_params
        )

        # THEN it should return valid results
        valid_result = (83.33333333333333, "83 %", 2.0, "")

        self.assertEqual(result, valid_result)

    def test_update_progress_bars_no_json(self):
        # GIVEN no run_log_json
        # WHEN update_progress_bars() is ran with that input
        result = update_progress_bars({}, model_params)

        # THEN it should return valid results
        valid_result = (0, 0, 0, 0)

        self.assertEqual(result, valid_result)


class TestProgressDisplay(unittest.TestCase):
    def test_update_progress_display_no_validation(self):
        # GIVEN known inputs
        # WHEN update_progress_display() is ran with that input
        result = update_progress_display(
            json.dumps(run_log_json_no_validation), model_params
        )

        # THEN it should return valid results
        valid_result = [
            html.P("Batch: 110 / 1000"),
            html.P("Epoch: 1 / 10"),
            html.P("Tracking precision: 0.01"),
        ]

        self.assertEqual(
            " ".join(result.children[0].children.split()), valid_result[0].children
        )
        self.assertEqual(result.children[1].children, valid_result[1].children)
        self.assertEqual(result.children[2].children, valid_result[2].children)

    def test_update_progress_display_with_validation(self):
        # GIVEN known inputs
        # WHEN update_progress_display() is ran with that input
        result = update_progress_display(
            json.dumps(run_log_json_with_validation), model_params
        )

        # THEN it should return valid results
        valid_result = [
            html.P("Batch: 850 / 1000"),
            html.P("Epoch: 2 / 10"),
            html.P("Tracking precision: 0.01"),
        ]

        self.assertEqual(
            " ".join(result.children[0].children.split()), valid_result[0].children
        )
        self.assertEqual(result.children[1].children, valid_result[1].children)
        self.assertEqual(
            " ".join(result.children[2].children.split()), valid_result[2].children
        )


class TestModelSummaryDivs(unittest.TestCase):
    def test_no_model_params(self):
        project_name = "my_project"
        result = get_model_summary_divs({}, model_summary, project_name)
        self.assertEqual(result, None)

    def test_no_model_summary(self):
        project_name = "my_project"
        result = get_model_summary_divs(model_params, {}, project_name)
        self.assertEqual(result, None)

    def test_model_summary(self):
        # GIVEN known inputs
        # WHEN get_model_summary_divs() is ran with that input
        project_name = "my_project"
        result = get_model_summary_divs(model_params, model_summary, project_name)

        valid_result = (
            html.Div(
                children=[
                    html.P("Number of layers:"),
                    html.P(3),
                    html.P("Total params:"),
                    html.P(101770),
                ],
                className="model-summary",
            ),
            html.Div(
                children=[
                    html.P("Project name:"),
                    html.P(project_name),
                    html.P("Model:"),
                    html.P("Type: Sequential"),
                    html.P("Name: sequential"),
                ],
                className="model-summary",
            ),
            html.Div(
                children=[
                    html.P("Input shape:"),
                    html.P("[None, 28, 28]"),
                    html.P("Output:"),
                    html.P("Units: 10"),
                    html.P("Activation: linear"),
                ],
                className="model-summary",
            ),
        )

        for i in range(len(result)):
            for j in range(len(result[i].children)):
                self.assertEqual(
                    result[i].children[j].children, valid_result[i].children[j].children
                )
