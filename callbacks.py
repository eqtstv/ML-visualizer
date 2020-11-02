import json
import pathlib
import pandas as pd
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html

from app import app
from dash.dependencies import Input, Output

LOGS_PATH = pathlib.Path(__file__).parent.joinpath("logs").resolve()

LOGFILE_TRAIN = "run_log_train.csv"
LOGFILE_VAL = "run_log_val.csv"
MODEL_SUMMARY = "model_summary.txt"
MODEL_PARAMS = "model_params.json"
EPOCH_TIMES = "epoch_times.csv"


def update_graph(
    graph_id,
    graph_title,
    y_train_index,
    y_val_index,
    run_log_json,
    yaxis_title,
):
    def smooth(scalars, weight=0.6):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    trace_train = go.Scatter()
    trace_val = go.Scatter()

    layout = go.Layout(
        template="plotly_dark",
        title_text=graph_title,
    )

    fig = go.Figure(data=[trace_train, trace_val], layout=layout)

    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")

        step = run_log_df["step"]
        y_train = run_log_df[y_train_index]

        if y_val_index in run_log_df:
            y_val = run_log_df[y_val_index]
        else:
            y_val = pd.Series()

        if not y_train.isnull().values.any():

            y_train = smooth(y_train)

            trace_train = go.Scatter(
                x=step,
                y=y_train,
                mode="lines",
                name="Training",
                showlegend=True,
            )

        if y_val.isnull().values.any():
            y_val = y_val.dropna()

            # y_val = smooth(y_val)

            trace_val = go.Scatter(
                x=y_val.index,
                y=y_val,
                mode="lines",
                name="Validation",
                showlegend=True,
            )

        fig = go.Figure(data=[trace_train, trace_val], layout=layout)

        fig.update_xaxes(range=[0, step.iloc[-1] * 1.1])
        fig.update_yaxes(
            range=[
                max(min(y_train[-10:-1]) - 0.1, -0.01),
                y_train[-1] + 0.1,
            ]
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=y_train[-1],
            x1=step.iloc[-1] * 1.1,
            y1=y_train[-1],
            line=dict(color="blue", dash="dot", width=1),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=0,
            y=y_train[-1],
            text=f"{y_train[-1]:.4f}",
            showarrow=False,
            yshift=11,
            xshift=22,
            font=dict(),
            bgcolor="rgb(50,50,150)",
        )

        if not y_val.empty:
            fig.update_yaxes(
                range=[
                    max(min(y_train[-1], y_val.iloc[-1]) - 0.1, -0.01),
                    min(max(y_train[-1], y_val.iloc[-1]) + 0.1, 1.01),
                ]
            )
            fig.add_shape(
                type="line",
                x0=0,
                y0=y_val.iloc[-1],
                x1=step.iloc[-1] * 1.1,
                y1=y_val.iloc[-1],
                line=dict(color="red", dash="dot", width=1),
                xref="x",
                yref="y",
            )
            fig.add_annotation(
                x=0,
                y=y_val.iloc[-1],
                text=f"{y_val.iloc[-1]:.4f}",
                showarrow=False,
                yshift=-11,
                xshift=22,
                font=dict(),
                bgcolor="rgb(150,50,50)",
            )

        return dcc.Graph(
            id=graph_id,
            config={
                "displayModeBar": False,
                "scrollZoom": True,
            },
            figure=fig,
        )
    return dcc.Graph(
        id=graph_id,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
        },
        figure=fig,
    )


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

    # Refreshes every 24 hours
    elif interval_rate == "no":
        return 24 * 60 * 60 * 1000


@app.callback(
    Output("model-stats-storage", "data"), [Input("interval-log-update", "n_intervals")]
)
def get_model_params(_):
    with open(f"{LOGS_PATH}/{MODEL_PARAMS}") as fp:
        data = json.load(fp)
        return data


@app.callback(
    Output("run-log-storage", "data"), [Input("interval-log-update", "n_intervals")]
)
def get_run_log(_):
    names_train = [
        "step",
        "batch",
        "train accuracy",
        "train loss",
    ]
    names_val = [
        "step",
        "val accuracy",
        "val loss",
        "epoch",
        "epoch time",
    ]

    try:
        df_train = pd.read_csv(f"{LOGS_PATH}/{LOGFILE_TRAIN}", names=names_train)
        df_val = pd.read_csv(f"{LOGS_PATH}/{LOGFILE_VAL}", names=names_val)
        run_log_df = pd.merge(df_train, df_val, on="step", how="left")
        json = run_log_df.to_json(orient="split")
    except FileNotFoundError as error:
        df_train = pd.read_csv(f"{LOGS_PATH}/{LOGFILE_TRAIN}", names=names_train)
        json = df_train.to_json(orient="split")
        return json

    return json


@app.callback(
    Output("div-step-display", "children"),
    [Input("run-log-storage", "data"), Input("model-stats-storage", "data")],
)
def update_div_step_display(run_log_json, model_stats):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")

        last_val_index = run_log_df["epoch"].last_valid_index()
        epoch = run_log_df["epoch"].iloc[last_val_index] + 1
        et = run_log_df["epoch time"].iloc[last_val_index]
        eta = et * model_stats["epochs"]

        return html.Div(
            children=[
                html.H6(f"Epoch: {epoch:.0f} / {model_stats['epochs']}"),
                html.H6(
                    f"Step: {run_log_df['step'].iloc[-1]} / {model_stats['no_tracked_steps']}"
                ),
                html.H6(f"Epoch time: {et:.4f} s."),
                html.H6(f"Estimated training time: {eta:.4f} s."),
            ]
        )


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
        "train accuracy",
        "val accuracy",
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
        "train loss",
        "val loss",
        run_log_json,
        "Loss",
    )

    return [graph]


@app.callback(
    Output("div-current-accuracy-value", "children"), [Input("run-log-storage", "data")]
)
def update_div_current_accuracy_value(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")
        last_val_index = run_log_df["epoch"].last_valid_index()
        return [
            html.P(
                "Current Accuracy:",
                style={
                    "font-weight": "bold",
                    "margin-top": "15px",
                    "margin-bottom": "0px",
                },
            ),
            html.Div(f"Training: {run_log_df['train accuracy'].iloc[-1]:.4f}"),
            html.Div(
                f"Validation: {run_log_df['val accuracy'].iloc[last_val_index]:.4f}"
            ),
        ]


@app.callback(
    Output("div-current-loss-value", "children"),
    [Input("run-log-storage", "data")],
)
def update_div_current_loss_value(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")
        last_val_index = run_log_df["epoch"].last_valid_index()
        return [
            html.P(
                "Current Loss:",
                style={
                    "font-weight": "bold",
                    "margin-top": "15px",
                    "margin-bottom": "0px",
                },
            ),
            html.Div(f"Training: {run_log_df['train loss'].iloc[-1]:.4f}"),
            html.Div(f"Validation: {run_log_df['val loss'].iloc[last_val_index]:.4f}"),
        ]


@app.callback(
    Output("div-model-summary", "children"), [Input("run-log-storage", "data")]
)
def get_model_summary(run_log_json):
    with open(f"{LOGS_PATH}/{MODEL_SUMMARY}") as fp:
        lines = []
        div = html.Div(children=lines)
        for line in fp:
            lines.append(html.P(line))

    return div


@app.callback(
    Output("div-model-params", "children"), [Input("model-stats-storage", "data")]
)
def get_model_params_div(model_stats):
    return html.Div(str(model_stats))


@app.callback(
    [
        Output("epoch-progress", "value"),
        Output("epoch-progress", "children"),
        Output("learning-progress", "value"),
        Output("learning-progress", "children"),
    ],
    [Input("run-log-storage", "data"), Input("model-stats-storage", "data")],
)
def update_progress(run_log_json, model_stats):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")

        batch_prog = run_log_df["batch"].iloc[-1] * 100 / model_stats["max_batch_step"]
        step_prog = run_log_df["step"].iloc[-1] * 100 / model_stats["no_tracked_steps"]

        return (
            batch_prog,
            f"{batch_prog:.0f} %" if step_prog >= 5 else "",
            step_prog,
            f"{step_prog:.0f} %" if step_prog >= 5 else "",
        )

    return 0, 0, 0, 0
