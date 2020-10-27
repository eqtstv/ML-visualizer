import dash
import pathlib
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

from dash.dependencies import Input, Output
from demo_utils import demo_callbacks, demo_explanation


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
demo_mode = True

DATA_PATH = pathlib.Path(__file__).parent.joinpath("data").resolve()

LOGFILE = "examples/run_log.csv"


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

    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient="split")

        step = run_log_df["step"]
        y_train = run_log_df[y_train_index]
        y_val = run_log_df[y_val_index]

        y_train = smooth(y_train)
        y_val = smooth(y_val)

        trace_train = go.Scatter(
            x=step,
            y=y_train,
            mode="lines",
            name="Training",
            showlegend=True,
        )

        trace_val = go.Scatter(
            x=step,
            y=y_val,
            mode="lines",
            name="Validation",
            showlegend=True,
        )

        layout = go.Layout(template="plotly_dark", title_text=graph_title)
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
            dcc.Store(id="storage-simulated-run", storage_type="memory"),
            dcc.Interval(
                id="interval-simulated-step",
                interval=125,  # Updates every 100 milliseconds, i.e. every step takes 25 ms
                n_intervals=0,
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        html.H1(
                            "ML Visualizer",
                            id="title",
                            style={"margin-left": "3%"},
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


demo_callbacks(app, demo_mode)


@app.callback(
    Output("div-accuracy-graph", "children"),
    [
        Input("run-log-storage", "data"),
    ],
)
def update_accuracy_graph(run_log_json):
    graph = update_graph(
        "accuracy-graph",
        "Prediction Accuracy",
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
        "cross-entropy-graph",
        "Cross Entropy Loss",
        "train cross entropy",
        "val cross entropy",
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
def update_div_current_cross_entropy_value(run_log_json):
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
            html.Div(f"Training: {run_log_df['train cross entropy'].iloc[-1]:.4f}"),
            html.Div(f"Validation: {run_log_df['val cross entropy'].iloc[-1]:.4f}"),
        ]


if __name__ == "__main__":
    app.run_server(debug=True)
