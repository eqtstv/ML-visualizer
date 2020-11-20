import json
import pathlib

from flask import Flask
from flask_jwt_extended import JWTManager
from flask_restful import Api

from ml_visualizer.database.database import init_db
from ml_visualizer.api.resources import (
    ClearData,
    ModelLayers,
    ModelParams,
    ModelSummary,
    TrainingLog,
    ValidationLog,
    Project,
)
from ml_visualizer.auth.api_auth import Auth
from ml_visualizer.database.models import User
from ml_visualizer.user.resources import Login, Logout, Signup, login_manager
from ml_visualizer.web.resources import DashApp, Index, Main, NotLogged, Profile


CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("config.json").resolve()
with open(CONFIG_PATH) as f:
    config = json.load(f)

server = Flask(__name__)
server.secret_key = config["secret_key"]

api = Api(server)
init_db()


login_manager.init_app(server)
jwt = JWTManager(server)


api.add_resource(Index, "/")
api.add_resource(Main, "/main")
api.add_resource(Login, "/login")
api.add_resource(Signup, "/signup")
api.add_resource(Logout, "/logout")
api.add_resource(Profile, "/profile")
api.add_resource(NotLogged, "/notlogged")
api.add_resource(DashApp, "/dashapp")

api.add_resource(Auth, "/auth")

api.add_resource(ClearData, "/clear")
api.add_resource(ModelParams, "/params")
api.add_resource(ModelSummary, "/summary")
api.add_resource(ModelLayers, "/layers")
api.add_resource(TrainingLog, "/train")
api.add_resource(ValidationLog, "/val")
api.add_resource(Project, "/project")
