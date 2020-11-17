import sys

from flask import jsonify, request
from flask_jwt_extended import jwt_required
from flask_restful import Resource


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
