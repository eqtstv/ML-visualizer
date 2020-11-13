import json
import unittest

import pandas as pd
import plotly
from ml_visualizer.callbacks_utils import (
    get_input_layer_info,
    get_layers,
    update_current_value,
    update_graph,
)

run_log_json = {
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
    "index": [0, 1, 75],
    "data": [
        [0, 0, 0.1875, 2.5382819176, None, None, None, None],
        [1, 10, 0.2, 2.2, None, None, None, None],
        [75, 0, 0.90625, 0.2020742446, 0.836499989, 0.4435300827, 1.0, 19.0966033],
    ],
}


class TestUpdateGraph(unittest.TestCase):
    def test_graph_names(self):
        # GIVEN graph with known parameters
        graph_params = [
            "example-graph",
            "Example Graph",
            "train_accuracy",
            "train_val",
            {},
            "Example_axis",
        ]

        # WHEN graph is initiated with that parameters
        graph = update_graph(*graph_params)

        # THEN grph should have proper name values
        self.assertEqual(graph.id, "example-graph")
        self.assertEqual(graph.figure.layout.title.text, "Example Graph")

    def test_graph_json(self):
        # GIVEN graph with known parameters
        graph_params = [
            "example-graph",
            "Example Graph",
            "train_accuracy",
            "train_val",
            json.dumps(run_log_json),
            "Example_axis",
        ]

        # WHEN graph is initiated with that parameters
        graph = update_graph(*graph_params)

        self.assertIsInstance(graph.figure.data[0], plotly.graph_objs._scatter.Scatter)
        self.assertIsInstance(graph.figure.data[1], plotly.graph_objs._scatter.Scatter)
