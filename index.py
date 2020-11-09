import json
import pathlib
import sys

from flask import Flask, jsonify, request
from flask_restful import Resource

import callbacks
from app import api, app, config, server
from database.database import Base, db_session
from database.models import LogTraining, LogValidation
from layout import app_layout

app.layout = app_layout
LOGS_PATH = f"{pathlib.Path(__file__).parent.resolve()}/{config['logs_folder']}"


def clear_data(session):
    meta = Base.metadata
    for table in reversed(meta.sorted_tables):
        print(f"Clear table {table}")
        session.execute(table.delete())
    session.commit()


@api.resource("/params")
class TodoSimple(Resource):
    data = {}
    filename = config["filenames"]["model_params"]

    def get(self):
        print("GET")
        return self.data

    def put(self):
        try:
            clear_data(db_session)
            print("PUT")
            self.data.update(request.json)
            print(self.data)

            with open(f"{LOGS_PATH}/{self.filename}", "a+", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
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
    app.run_server(host="192.168.0.158", port="5050", debug=True)
