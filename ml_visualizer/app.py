import json
import pathlib

import dash
import dash_bootstrap_components as dbc
from flask import Flask
from flask_restful import Api

from ml_visualizer.auth.database.database import init_users_db, users_db
from ml_visualizer.auth.database.models import User
from ml_visualizer.auth.resources import (
    Index,
    Login,
    Logout,
    Main,
    NotLogged,
    Profile,
    Signup,
    login_manager,
)
from ml_visualizer.database.database import init_db
from ml_visualizer.resources.data import ClearData
from ml_visualizer.resources.logs import TrainingLog, ValidationLog
from ml_visualizer.resources.model import ModelLayers, ModelParams, ModelSummary

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("config.json").resolve()
with open(CONFIG_PATH) as f:
    config = json.load(f)

server = Flask("my_app")
server.secret_key = config["secret_key"]

api = Api(server)
init_db()
init_users_db()


login_manager.init_app(server)


api.add_resource(Index, "/")
api.add_resource(Main, "/main")
api.add_resource(Login, "/login")
api.add_resource(Signup, "/signup")
api.add_resource(Logout, "/logout")
api.add_resource(Profile, "/profile")
api.add_resource(NotLogged, "/notlogged")

api.add_resource(ClearData, "/clear")
api.add_resource(ModelParams, "/params")
api.add_resource(ModelSummary, "/summary")
api.add_resource(ModelLayers, "/layers")
api.add_resource(TrainingLog, "/train")
api.add_resource(ValidationLog, "/val")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)
