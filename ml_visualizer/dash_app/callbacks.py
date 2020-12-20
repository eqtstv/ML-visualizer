import pandas as pd
import requests
from dash.dependencies import Input, Output
from flask import request
from flask_login import current_user
from ml_visualizer.dash_app.callbacks_utils import (
    get_model_summary_divs,
    update_current_value,
    update_graph,
    update_interval_log,
    update_progress_bars,
    update_progress_display,
)
from ml_visualizer.dash_app.dash_app import dash_app
from ml_visualizer.database import engine


@dash_app.callback(
    Output("interval-log-update", "interval"),
    [Input("dropdown-interval-control", "value")],
)
def update_interval_log_update(interval_rate):
    return update_interval_log(interval_rate)


@dash_app.callback(
    Output("current-project", "data"),
    [Input("interval-log-update", "n_intervals")],
)
def get_current_project(_):
    try:
        return requests.get(f"{request.url_root}/current_project").json()[
            "current_project"
        ]
    except Exception as e:
        print(e)


@dash_app.callback(
    Output("model-params-storage", "data"),
    [Input("interval-log-update", "n_intervals"), Input("current-project", "data")],
)
def get_model_params(_, current_project):
    try:
        with engine.connect() as connection:
            model_params = pd.read_sql(
                f"SELECT \
                tracking_precision, \
                no_steps, \
                epochs, \
                batch_split, \
                max_batch_step, \
                steps_in_batch, \
                no_tracked_steps, \
                total_params \
                FROM model_params \
                WHERE user_id=={current_user.id} \
                AND project_name=='{current_project}'",
                connection,
            )
        return model_params.to_dict()
    except Exception as e:
        print(e)


@dash_app.callback(
    Output("model-summary-storage", "data"),
    [Input("interval-log-update", "n_intervals"), Input("current-project", "data")],
)
def get_model_summary(_, current_project):
    try:
        with engine.connect() as connection:
            model_summary = pd.read_sql(
                f"SELECT \
                class_name, \
                config \
                FROM model_summary \
                WHERE user_id=={current_user.id} \
                AND project_name=='{current_project}'",
                connection,
            )
        return model_summary.to_dict()
    except Exception as e:
        print(e)
        return None


@dash_app.callback(
    Output("run-log-storage", "data"),
    [Input("interval-log-update", "n_intervals"), Input("current-project", "data")],
)
def get_run_log(_, current_project):
    try:
        with engine.connect() as connection:
            df_train = pd.read_sql(
                f"SELECT step, batch, train_accuracy, train_loss \
                  FROM log_training WHERE user_id=={current_user.id} \
                  AND project_name=='{current_project}'",
                connection,
            )
            df_val = pd.read_sql(
                f"SELECT step, val_accuracy, val_loss, epoch, epoch_time \
                  FROM log_validation WHERE user_id=={current_user.id} \
                  AND project_name=='{current_project}'",
                connection,
            )

        run_log_df = pd.merge(df_train, df_val, on="step", how="left")
        json = run_log_df.to_json(orient="split")
        return json
    except Exception as e:
        print(e)

    try:
        with engine.connect() as connection:
            df_train = pd.read_sql(
                f"SELECT step, batch, train_accuracy, train_loss \
                  FROM log_training WHERE user_id=={current_user.id} \
                  AND project_name=='{current_project}'",
                connection,
            )
        json = df_train.to_json(orient="split")
        return json
    except Exception as e:
        print(e)


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
    [
        Input("model-params-storage", "data"),
        Input("model-summary-storage", "data"),
        Input("current-project", "data"),
    ],
)
def update_model_summary_divs(model_params, model_summary, current_project):
    return get_model_summary_divs(model_params, model_summary, current_project)


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
