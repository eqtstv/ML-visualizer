import json
import pathlib

import dash
import dash_bootstrap_components as dbc

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("config.json").resolve()

with open(CONFIG_PATH) as f:
    config = json.load(f)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
