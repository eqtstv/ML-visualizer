import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

df = pd.read_csv("data/stockdata2.csv")
df1 = df.loc[df["stock"] == "AAPL"]
df2 = df.loc[df["stock"] == "MSFT"]


def update_graph(df, graph_id, graph_title):

    trace_train = go.Scatter(
        x=df1["Date"],
        y=df1["value"],
        mode="lines",
        name="Training",
        showlegend=True,
    )

    trace_val = go.Scatter(
        x=df2["Date"],
        y=df2["value"],
        mode="lines",
        name="Validation",
        showlegend=True,
    )

    layout = go.Layout(template="plotly_dark", title_text=graph_title)
    figure = go.Figure(data=[trace_train, trace_val], layout=layout)

    return dcc.Graph(
        id=graph_id,
        className="chart-graph",
        config={
            "displayModeBar": False,
            "scrollZoom": True,
        },
        figure=figure,
    )


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
                                    update_graph(df1, "graph1", "AAPL"),
                                    update_graph(df2, "graph2", "Microsoft"),
                                    html.Div(
                                        children=[
                                            html.P(
                                                f"Training: {df1.value.iloc[-1]:.4f}"
                                            ),
                                            html.P(
                                                f"Validation: {df1.value.iloc[1]:.4f}"
                                            ),
                                        ],
                                        className="chart-graph",
                                    ),
                                    html.Div(
                                        children=[
                                            html.P(
                                                f"Training: {df2.value.iloc[-1]:.4f}"
                                            ),
                                            html.P(
                                                f"Validation: {df2.value.iloc[1]:.4f}"
                                            ),
                                        ],
                                        className="chart-graph",
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
