import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

df = pd.DataFrame(
    {
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"],
    }
)

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
fig["layout"]["template"] = "plotly_dark"

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
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        id="example-graph-2",
                                        className="chart-graph",
                                        config={
                                            "displayModeBar": False,
                                            "scrollZoom": True,
                                        },
                                        figure=fig,
                                    ),
                                    dcc.Graph(
                                        id="example-graph-3",
                                        className="chart-graph",
                                        config={
                                            "displayModeBar": False,
                                            "scrollZoom": True,
                                        },
                                        figure=fig,
                                    ),
                                ],
                                className="graphs-frame",
                            ),
                        ),
                        className="main-frame",
                    ),
                ],
                style={"height": "92%"},
            ),
        ],
    ),
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(debug=True)
