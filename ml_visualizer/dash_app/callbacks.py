import dash_html_components as html
import pandas as pd
import requests
from dash.dependencies import Input, Output
from ml_visualizer.api.database import engine
from ml_visualizer.app import config
from ml_visualizer.dash_app.dash_app import dash_app
from ml_visualizer.dash_app.callbacks_utils import (
    get_model_summary_divs,
    update_current_value,
    update_graph,
    update_interval_log,
    update_progress_bars,
    update_progress_display,
)

URL = f"http://{config['ip']}:{config['port']}"


@dash_app.callback(
    Output("interval-log-update", "interval"),
    [Input("dropdown-interval-control", "value")],
)
def update_interval_log_update(interval_rate):
    return update_interval_log(interval_rate)


@dash_app.callback(
    Output("model-params-storage", "data"),
    [Input("interval-log-update", "n_intervals")],
)
def get_model_params(_):
    try:
        return requests.get(f"{URL}/params").json()
    except:
        return None


@dash_app.callback(
    Output("model-summary-storage", "data"),
    [Input("interval-log-update", "n_intervals")],
)
def get_model_params(_):
    try:
        return requests.get(f"{URL}/summary").json()
    except:
        return None


@dash_app.callback(
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


@dash_app.callback(
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


@dash_app.callback(
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


@dash_app.callback(
    Output("div-current-accuracy-value", "children"), [Input("run-log-storage", "data")]
)
def update_div_current_accuracy_value(run_log_json):
    return update_current_value(
        "train_accuracy", "val_accuracy", "Accuracy", run_log_json
    )


@dash_app.callback(
    Output("div-current-loss-value", "children"),
    [Input("run-log-storage", "data")],
)
def update_div_current_loss_value(run_log_json):
    return update_current_value("train_loss", "val_loss", "Loss", run_log_json)


@dash_app.callback(
    Output("div-model-summary", "children"),
    [Input("model-params-storage", "data"), Input("model-summary-storage", "data")],
)
def update_model_summary_divs(model_params, model_summary):
    return get_model_summary_divs(model_params, model_summary)


@dash_app.callback(
    Output("div-model-params", "children"), [Input("model-params-storage", "data")]
)
def get_model_params_div(model_params):
    pass
    # return html.Div(html.P(str(model_params)))


@dash_app.callback(
    Output("div-epoch-step-display", "children"),
    [Input("run-log-storage", "data"), Input("model-params-storage", "data")],
)
def update_div_step_display(run_log_json, model_params):
    return update_progress_display(run_log_json, model_params)


@dash_app.callback(
    [
        Output("epoch-progress", "value"),
        Output("epoch-progress", "children"),
        Output("learning-progress", "value"),
        Output("learning-progress", "children"),
    ],
    [Input("run-log-storage", "data"), Input("model-params-storage", "data")],
)
def update_progress(run_log_json, model_params):
    return update_progress_bars(run_log_json, model_params)