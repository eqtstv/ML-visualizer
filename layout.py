import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


app_layout = dbc.Container(
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
                style={"height": "6%"},
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
                        width=2,
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
                                            children=[
                                                html.Div(
                                                    id="div-current-accuracy-value",
                                                    className="chart-stats",
                                                ),
                                                html.Div(
                                                    className="chart-stats",
                                                    id="div-interval-control",
                                                    children=[
                                                        html.Div(
                                                            id="div-step-display",
                                                        ),
                                                        html.Div(
                                                            children=[
                                                                dcc.Interval(
                                                                    id="progress-interval",
                                                                    n_intervals=0,
                                                                    interval=500,
                                                                ),
                                                                dbc.Progress(
                                                                    id="epoch-progress",
                                                                    className="mb-3",
                                                                    striped=True,
                                                                    animated=True,
                                                                ),
                                                                dbc.Progress(
                                                                    id="learning-progress",
                                                                    className="mb-3",
                                                                    striped=True,
                                                                    animated=True,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="div-current-loss-value",
                                                    className="chart-stats",
                                                ),
                                                html.Div(id="div-model-summary"),
                                                html.Div(id="div-model-params"),
                                            ],
                                            className="div-graphs-stats",
                                        ),
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
                style={"height": "92%"},
            ),
            dcc.Interval(id="interval-log-update", n_intervals=0),
            dcc.Store(id="run-log-storage", storage_type="memory"),
            dcc.Store(id="model-stats-storage", storage_type="memory"),
        ],
    ),
    fluid=True,
)