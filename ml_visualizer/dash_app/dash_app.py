print("starting dash")
import dash
import dash_bootstrap_components as dbc
from ml_visualizer.app import server

from ml_visualizer.dash_app.layout import app_layout


dash_app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    server=server,
    url_base_pathname="/dash_app/",
)

dash_app.layout = app_layout
