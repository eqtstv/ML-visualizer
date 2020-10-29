import dash
import pathlib
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

from dash.dependencies import Input, Output

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


LOGS_PATH = pathlib.Path(__file__).parent.joinpath("logs").resolve()

LOGFILE_TRAIN = "run_log_train.csv"
LOGFILE_VAL = "run_log_val.csv"


def update_graph(
    graph_id,
    graph_title,
    y_train_index,
    y_val_index,
    run_log_json,
    yaxis_title,
):
    def smooth(scalars, weight=0.5):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")

        step = run_log_df["step"]
        y_train = run_log_df[y_train_index]
        y_val = run_log_df[y_val_index]

        trace_train = go.Scatter()
        trace_val = go.Scatter()

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

        layout = go.Layout(
            template="plotly_dark",
            title_text=graph_title,
        )
        figure = go.Figure(data=[trace_train, trace_val], layout=layout)

        return dcc.Graph(
            id=graph_id,
            config={
                "displayModeBar": False,
                "scrollZoom": True,
            },
            figure=figure,
        )
    return dcc.Graph(id=graph_id)


app.layout = dbc.Container(
    html.Div(
        className="main-app",
        children=[
            dbc.Row(
                dbc.Col(
                    html.Div(
                        html.H1(
                            "ML Visualizer",
                            id="title",
                        ),
                    ),
                    className="banner",
                ),
                style={"height": "8%"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            children=[
                                html.H2("Main"),
                                html.H2("Explore data"),
                                html.H2("Graphs"),
                            ],
                        ),
                        className="side-bar",
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Row(
                            [
                                html.Div(
                                    children=[
                                        html.Div(
                                            id=f"div-accuracy-graph",
                                            className="chart-graph",
                                        ),
                                        html.Div(
                                            id=f"div-loss-graph",
                                            className="chart-graph",
                                        ),
                                        html.Div(
                                            id="div-current-accuracy-value",
                                            className="chart-stats",
                                        ),
                                        html.Div(
                                            id="div-current-loss-value",
                                            className="chart-stats",
                                        ),
                                    ],
                                    className="graphs-frame",
                                ),
                                dcc.Dropdown(
                                    id="dropdown-demo-dataset",
                                    options=[
                                        {
                                            "label": "CIFAR 10",
                                            "value": "cifar",
                                        },
                                        {
                                            "label": "MNIST",
                                            "value": "mnist",
                                        },
                                        {
                                            "label": "Fashion MNIST",
                                            "value": "fashion",
                                        },
                                    ],
                                    value="mnist",
                                    placeholder="Select a demo dataset",
                                    searchable=False,
                                ),
                                html.Div(
                                    dcc.Dropdown(
                                        id="dropdown-simulation-model",
                                        options=[
                                            {
                                                "label": "1-Layer Neural Net",
                                                "value": "softmax",
                                            },
                                            {
                                                "label": "Simple Conv Net",
                                                "value": "cnn",
                                            },
                                        ],
                                        value="cnn",
                                        placeholder="Select Model to Simulate",
                                        searchable=False,
                                    ),
                                    className="six columns dropdown-box-second",
                                ),
                                html.Div(
                                    dcc.Dropdown(
                                        id="dropdown-interval-control",
                                        options=[
                                            {
                                                "label": "No Updates",
                                                "value": "no",
                                            },
                                            {
                                                "label": "Slow Updates",
                                                "value": "slow",
                                            },
                                            {
                                                "label": "Regular Updates",
                                                "value": "regular",
                                            },
                                            {
                                                "label": "Fast Updates",
                                                "value": "fast",
                                            },
                                        ],
                                        value="regular",
                                        className="twelve columns dropdown-box-third",
                                        clearable=False,
                                        searchable=False,
                                    )
                                ),
                                html.Div(
                                    className="four columns",
                                    id="div-interval-control",
                                    children=[
                                        html.Div(
                                            id="div-total-step-count",
                                            className="twelve columns",
                                        ),
                                        html.Div(
                                            id="div-step-display",
                                            className="twelve columns",
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        className="main-frame",
                    ),
                ],
                style={"height": "92%"},
            ),
            dcc.Interval(id="interval-log-update", n_intervals=0),
            dcc.Store(id="run-log-storage", storage_type="memory"),
        ],
    ),
    fluid=True,
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
    Output("run-log-storage", "data"), [Input("interval-log-update", "n_intervals")]
)
def get_run_log(_):
    names_train = [
        "step",
        "train accuracy",
        "train loss",
    ]
    names_val = [
        "step",
        "val accuracy",
        "val loss",
    ]

    try:
        df_train = pd.read_csv(f"{LOGS_PATH}/{LOGFILE_TRAIN}", names=names_train)
        df_val = pd.read_csv(f"{LOGS_PATH}/{LOGFILE_VAL}", names=names_val)
        run_log_df = pd.merge(df_train, df_val, on="step", how="left")
        json = run_log_df.to_json(orient="split")
    except FileNotFoundError as error:
        print(error)
        print(
            "Please verify if the csv file generated by your model is placed in the correct directory."
        )
        return None

    return json


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
            html.Div(f"Validation: {run_log_df['val accuracy'].iloc[-1]:.4f}"),
        ]


@app.callback(
    Output("div-current-loss-value", "children"),
    [Input("run-log-storage", "data")],
)
def update_div_current_loss_value(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")
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
            html.Div(f"Validation: {run_log_df['val loss'].iloc[-1]:.4f}"),
        ]


if __name__ == "__main__":
    app.run_server(debug=True)
