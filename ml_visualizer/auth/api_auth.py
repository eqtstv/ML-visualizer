from flask import jsonify, make_response, request
from flask_jwt_extended import create_access_token
from flask_restful import Resource
from ml_visualizer.user.models import User
from werkzeug.security import check_password_hash


class Auth(Resource):
    def post(self):
        if not request.is_json:
            return make_response(jsonify({"msg": "Missing JSON in request"}), 400)

        email = request.json.get("email", None)
        password = request.json.get("password", None)

        if not email:
            return make_response(jsonify({"msg": "Missing email parameter"}), 400)
        if not password:
            return make_response(jsonify({"msg": "Missing password parameter"}), 400)

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password_hash, password):
            return make_response(jsonify({"msg": "Bad email or password"}), 401)

        access_token = create_access_token(identity=email)
        return make_response(jsonify(access_token=access_token), 200)
