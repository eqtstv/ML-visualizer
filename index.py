import json
import pathlib
import sys

from flask import Flask, jsonify, request
from flask_restful import Resource

import ml_visualizer.callbacks
from ml_visualizer.app import api, app, config, server
from ml_visualizer.database.database import Base, db_session
from ml_visualizer.database.models import LogTraining, LogValidation
from ml_visualizer.layout import app_layout

app.layout = app_layout


@api.resource("/clear")
class ClearData(Resource):
    def delete(self):
        meta = Base.metadata
        for table in reversed(meta.sorted_tables):
            db_session.execute(table.delete())
        db_session.commit()


@api.resource("/params")
class ModelParams(Resource):
    data = {}

    def get(self):
        return self.data

    def put(self):
        try:
            self.data.update(request.json)
        except:
            e = sys.exc_info()
            print(e)

        return jsonify(self.data)


@api.resource("/summary")
class ModelSummary(Resource):
    data = {}

    def get(self):
        return self.data

    def put(self):
        try:
            self.data.update(request.json)
        except:
            e = sys.exc_info()
            print(e)

        return jsonify(self.data)


@api.resource("/layers")
class ModelLayers(Resource):
    data = {}

    def get(self):
        return self.data

    def put(self):
        try:
            self.data.update(request.json)
        except:
            e = sys.exc_info()
            print(e)

        return jsonify(self.data)


@api.resource("/train")
class TrainingLog(Resource):
    def put(self):
        try:
            data = request.json
            train_log = LogTraining(
                data["step"],
                data["batch"],
                data["train_loss"],
                data["train_accuracy"],
            )
            db_session.add(train_log)
            db_session.commit()

        except:
            e = sys.exc_info()
            print(e)


@api.resource("/val")
class ValidationLog(Resource):
    def put(self):
        try:
            data = request.json
            val_log = LogValidation(
                data["step"],
                data["val_loss"],
                data["val_accuracy"],
                data["epoch"],
                data["epoch_time"],
            )
            db_session.add(val_log)
            db_session.commit()

        except:
            e = sys.exc_info()
            print(e)


if __name__ == "__main__":
    app.run_server(host=f"{config['ip']}", port=f"{config['port']}", debug=True)
