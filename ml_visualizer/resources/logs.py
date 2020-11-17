import sys

from flask import jsonify, request
from flask_jwt_extended import jwt_required
from flask_restful import Resource
from ml_visualizer.database.database import Base, db_session
from ml_visualizer.database.models import LogTraining, LogValidation


class TrainingLog(Resource):
    @jwt_required
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


class ValidationLog(Resource):
    @jwt_required
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
