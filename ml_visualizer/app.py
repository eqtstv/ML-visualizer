import json
import pathlib
from flask import Flask, request
from flask_restful import Resource, Api
from ml_visualizer.database.database import init_db

import dash
import dash_bootstrap_components as dbc

server = Flask("my_app")
api = Api(server)
init_db()

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("config.json").resolve()

with open(CONFIG_PATH) as f:
    config = json.load(f)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)
