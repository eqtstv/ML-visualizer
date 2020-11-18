from flask import make_response, render_template
from flask_restful import Resource
from flask_login import current_user, login_required


class Index(Resource):
    def get(self):
        return make_response(
            render_template("index.html", user=current_user),
            200,
            {"Content-Type": "text/html"},
        )


class Main(Resource):
    def get(self):
        return make_response(
            render_template("index.html", user=current_user),
            200,
            {"Content-Type": "text/html"},
        )


class Profile(Resource):
    @login_required
    def get(self):
        return make_response(
            render_template("profile.html", user=current_user),
            200,
            {"Content-Type": "text/html"},
        )


class NotLogged(Resource):
    def get(self):
        return make_response(
            render_template("not_logged.html", user=current_user),
            200,
            {"Content-Type": "text/html"},
        )


class DashApp(Resource):
    @login_required
    def get(self):
        return make_response(
            render_template("dash_app.html", user=current_user),
            200,
            {"Content-Type": "text/html"},
        )
