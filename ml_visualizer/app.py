import json
import pathlib

import dash
import dash_bootstrap_components as dbc
from flask import Flask
from flask_restful import Api

from ml_visualizer.database.database import init_db
from ml_visualizer.resources.data import ClearData
from ml_visualizer.resources.logs import TrainingLog, ValidationLog
from ml_visualizer.resources.model import ModelLayers, ModelParams, ModelSummary

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("config.json").resolve()
server = Flask("my_app")
api = Api(server)
init_db()

api.add_resource(ClearData, "/clear")
api.add_resource(ModelParams, "/params")
api.add_resource(ModelSummary, "/summary")
api.add_resource(ModelLayers, "/layers")
api.add_resource(TrainingLog, "/train")
api.add_resource(ValidationLog, "/val")

with open(CONFIG_PATH) as f:
    config = json.load(f)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)
