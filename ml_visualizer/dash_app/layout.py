import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

app_layout = dbc.Container(
    html.Div(
        className="main-app",
        children=[
            dcc.Interval(id="interval-log-update", n_intervals=0),
            dcc.Store(id="run-log-storage", storage_type="memory"),
            dcc.Store(id="model-params-storage", storage_type="memory"),
            dcc.Store(id="model-summary-storage", storage_type="memory"),
            dcc.Store(id="current-project", storage_type="memory"),
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
                                    children=[
                                        html.Div(
                                            children=[
                                                dcc.Interval(
                                                    id="progress-interval",
                                                    n_intervals=0,
                                                    interval=500,
                                                ),
                                                dbc.Progress(
                                                    id="epoch-progress",
                                                    className="prog-bar",
                                                    striped=True,
                                                    animated=True,
                                                ),
                                                dbc.Progress(
                                                    id="learning-progress",
                                                    className="prog-bar",
                                                    striped=True,
                                                    animated=True,
                                                ),
                                            ],
                                            className="progress-bars-div",
                                        ),
                                        html.Div(
                                            id="div-current-accuracy-value",
                                            className="chart-stats",
                                        ),
                                        html.Div(
                                            className="chart-stats",
                                            id="div-interval-control",
                                            children=[
                                                html.Div(
                                                    id="div-epoch-step-display",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="div-current-loss-value",
                                            className="chart-stats",
                                        ),
                                    ],
                                    className="div-graphs-stats",
                                ),
                                html.Div(id="div-model-summary"),
                                html.Div(id="div-model-params"),
                            ],
                            className="graphs-frame",
                        ),
                        html.Div(
                            dcc.Dropdown(
                                id="dropdown-interval-control",
                                options=[
                                    {
                                        "label": "Fast Updates",
                                        "value": "fast",
                                    },
                                ],
                                value="fast",
                                style={"display": "none"},
                            )
                        ),
                    ]
                ),
                className="main-frame",
            ),
        ],
    ),
    fluid=True,
)