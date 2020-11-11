import sys

from ml_visualizer.app import api, app, config
from ml_visualizer.layout import app_layout
from ml_visualizer.resources.data import ClearData
from ml_visualizer.resources.logs import TrainingLog, ValidationLog
from ml_visualizer.resources.model import ModelLayers, ModelParams, ModelSummary

app.layout = app_layout

api.add_resource(ClearData, "/clear")
api.add_resource(ModelParams, "/params")
api.add_resource(ModelSummary, "/summary")
api.add_resource(ModelLayers, "/layers")
api.add_resource(TrainingLog, "/train")
api.add_resource(ValidationLog, "/val")


if __name__ == "__main__":
    app.run_server(host=f"{config['ip']}", port=f"{config['port']}", debug=True)
