import dash_html_components as html
import pandas as pd
import requests
from dash.dependencies import Input, Output

from ml_visualizer.app import app, config
from ml_visualizer.callbacks_utils import (
    get_input_layer_info,
    get_layers,
    update_current_value,
    update_graph,
)
from ml_visualizer.database.database import engine

URL = f"http://{config['ip']}:{config['port']}"


@app.callback(
    Output("interval-log-update", "interval"),
    [Input("dropdown-interval-control", "value")],
)
def update_interval_log_update(interval_rate):
    if interval_rate == "fast":
        return 500

    elif interval_rate == "regular":
        return 1000

    elif interval_rate == "slow":
        return 5 * 1000

    elif interval_rate == "no":
        return 24 * 60 * 60 * 1000


@app.callback(
    Output("model-params-storage", "data"),
    [Input("interval-log-update", "n_intervals")],
)
def get_model_params(_):
    try:
        return requests.get(f"{URL}/params").json()
    except:
        return None


@app.callback(
    Output("run-log-storage", "data"), [Input("interval-log-update", "n_intervals")]
)
def get_run_log(_):
    try:
        with engine.connect() as connection:
            df_train = pd.read_sql(
                "SELECT step, batch, train_accuracy, train_loss FROM log_training",
                connection,
            )
            df_val = pd.read_sql(
                "SELECT step, val_accuracy, val_loss, epoch, epoch_time FROM log_validation",
                connection,
            )

        run_log_df = pd.merge(df_train, df_val, on="step", how="left")
        json = run_log_df.to_json(orient="split")
        return json
    except:
        return None

    try:
        with engine.connect() as connection:
            df_train = pd.read_sql(
                "SELECT step, batch, train_accuracy, train_loss FROM log_training",
                connection,
            )
        json = df_train.to_json(orient="split")
        return json
    except:
        return None


@app.callback(
    Output("div-accuracy-graph", "children"),
    [
        Input("run-log-storage", "data"),
    ],
)
def update_accuracy_graph(run_log_json):
    graph = update_graph(
        "accuracy-graph",
        "Accuracy",
        "train_accuracy",
        "val_accuracy",
        run_log_json,
        "Accuracy",
    )
    return [graph]


@app.callback(
    Output("div-loss-graph", "children"),
    [
        Input("run-log-storage", "data"),
    ],
)
def update_loss_graph(run_log_json):
    graph = update_graph(
        "loss-graph",
        "Loss",
        "train_loss",
        "val_loss",
        run_log_json,
        "Loss",
    )

    return [graph]


@app.callback(
    Output("div-current-accuracy-value", "children"), [Input("run-log-storage", "data")]
)
def update_div_current_accuracy_value(run_log_json):
    return update_current_value(
        "train_accuracy", "val_accuracy", "Accuracy", run_log_json
    )


@app.callback(
    Output("div-current-loss-value", "children"),
    [Input("run-log-storage", "data")],
)
def update_div_current_loss_value(run_log_json):
    return update_current_value("train_loss", "val_loss", "Loss", run_log_json)


@app.callback(
    Output("div-model-summary", "children"),
    [Input("run-log-storage", "data"), Input("model-params-storage", "data")],
)
def get_model_summary(run_log_json, model_stats):
    model_summary = requests.get(f"{URL}/summary").json()

    if model_summary and model_stats:
        input_layer_info = get_input_layer_info(model_summary)
        layers_info = get_layers(model_summary)

        model_class_name_div = html.Div(
            children=[
                html.P("Model:"),
                html.P(f"Type: {model_summary['class_name']}"),
                html.P(f"Name: {model_summary['config']['name']}"),
            ],
            className="model-summary",
        )
        model_input_layer_info_div = html.Div(
            children=[
                html.P(f"Input shape:"),
                html.P(f"{input_layer_info['input_shape']}"),
                html.P(f"Output:"),
                html.P(f"Units: {layers_info[-1]['units']}"),
                html.P(f"Activation: {layers_info[-1]['activation']}"),
            ],
            className="model-summary",
        )
        model_layers_div = html.Div(
            children=[html.P("Layers:"), html.Div(layers_info)],
            className="model-summary",
        )

        model_layers_info = html.Div(
            children=[
                html.P("Number of layers:"),
                html.P(len(layers_info) - 1),
                html.P("Total params:"),
                html.P(model_stats["total_params"]),
            ],
            className="model-summary",
        )

        return model_class_name_div, model_layers_info, model_input_layer_info_div


@app.callback(
    Output("div-model-params", "children"), [Input("model-params-storage", "data")]
)
def get_model_params_div(model_stats):
    pass
    # return html.Div(html.P(str(model_stats)))


@app.callback(
    Output("div-epoch-step-display", "children"),
    [Input("run-log-storage", "data"), Input("model-params-storage", "data")],
)
def update_div_step_display(run_log_json, model_stats):
    steps_div = ()
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")
        if len(run_log_df["batch"]) != 0 and model_stats:
            residue = model_stats["no_steps"] - model_stats["max_batch_step"]
            if residue == 0:
                residue = model_stats["batch_split"]
            steps_div = (
                html.P(
                    f"Batch: {run_log_df['batch'].iloc[-1] + residue} / {model_stats['no_steps']}"
                ),
            )
            epochs_div = html.P(f"Epoch: {1:.0f} / {model_stats['epochs']}")
            tracking_precision = html.P(
                f"Tracking precision: {model_stats['tracking_precision']}"
            )

        if run_log_df["epoch"].last_valid_index() and model_stats:
            last_val_index = run_log_df["epoch"].last_valid_index()
            epoch = run_log_df["epoch"].iloc[last_val_index] + 1
            epochs_div = html.P(f"Epoch: {epoch:.0f} / {model_stats['epochs']}")

            et = run_log_df["epoch_time"].iloc[last_val_index]
            eta = et * model_stats["epochs"]
            epoch_time_div = html.P(f"Epoch time: {et:.4f} s.")
            eta_div = html.P(f"Estimated training time: {eta:.4f} s.")

            return html.Div(
                children=[
                    steps_div[0],
                    epochs_div,
                    tracking_precision,
                ],
                className="learning-stats",
            )
        if model_stats and len(steps_div) > 0:
            return html.Div(
                children=[steps_div[0], epochs_div, tracking_precision],
                className="learning-stats",
            )


@app.callback(
    [
        Output("epoch-progress", "value"),
        Output("epoch-progress", "children"),
        Output("learning-progress", "value"),
        Output("learning-progress", "children"),
    ],
    [Input("run-log-storage", "data"), Input("model-params-storage", "data")],
)
def update_progress(run_log_json, model_stats):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")
        if len(run_log_df["batch"]) != 0 and model_stats:
            batch_prog = (
                (run_log_df["batch"].iloc[-1]) * 100 / model_stats["max_batch_step"]
            )
            step_prog = (
                run_log_df["step"].iloc[-1] * 100 / model_stats["no_tracked_steps"]
            )

            return (
                batch_prog,
                f"{batch_prog:.0f} %" if batch_prog >= 5 else "",
                step_prog,
                f"{step_prog:.0f} %" if step_prog >= 5 else "",
            )

    return 0, 0, 0, 0
