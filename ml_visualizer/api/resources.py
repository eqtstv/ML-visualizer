import sys

from flask import jsonify, request, make_response
from flask_jwt_extended import jwt_required
from flask_restful import Resource
from ml_visualizer.api.database import Base, db_session
from ml_visualizer.api.models import LogTraining, LogValidation, Projects


class ModelParams(Resource):
    data = {}

    def get(self):
        return self.data

    @jwt_required
    def put(self):
        try:
            self.data.update(request.json)
        except:
            e = sys.exc_info()
            print(e)

        return jsonify(self.data)


class ModelSummary(Resource):
    data = {}

    def get(self):
        return self.data

    @jwt_required
    def put(self):
        try:
            self.data.update(request.json)
        except:
            e = sys.exc_info()
            print(e)

        return jsonify(self.data)


class ModelLayers(Resource):
    data = {}

    def get(self):
        return self.data

    @jwt_required
    def put(self):
        try:
            self.data.update(request.json)
        except:
            e = sys.exc_info()
            print(e)

        return jsonify(self.data)


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


class Project(Resource):
    @jwt_required
    def post(self):
        try:
            data = request.json
            selected_project = Projects.query.filter_by(
                project_name=data["project_name"]
            ).first()

            if selected_project:
                return make_response(jsonify({"is_valid": True}), 200)
            else:
                return make_response(
                    jsonify(
                        {
                            "msg": "Invalid project name!\nDo you want to create new project? (yes/no)"
                        }
                    ),
                    401,
                )
        except:
            e = sys.exc_info()
            print(e)

    @jwt_required
    def put(self):
        try:
            data = request.json
            selected_project = Projects.query.filter_by(
                project_name=data["project_name"]
            ).first()
            if selected_project:
                return make_response(
                    jsonify({"msg": "This project already exists."}),
                    401,
                )
            else:
                data = request.json
                new_project = Projects(
                    data["project_name"], data["project_description"]
                )
                db_session.add(new_project)
                db_session.commit()
        except:
            e = sys.exc_info()
            print(e)


class ClearData(Resource):
    @jwt_required
    def delete(self):
        meta = Base.metadata
        for table in reversed(meta.sorted_tables):

            if str(table.name) != "projects":
                db_session.execute(table.delete())
        db_session.commit()
