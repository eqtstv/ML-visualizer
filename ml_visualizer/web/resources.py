from flask import make_response, render_template
from flask_restful import Resource
from flask_login import current_user, login_required
from ml_visualizer.database.models import Projects


def get_all_user_projects():
    projects = Projects.query.filter_by(user_id=current_user.id)
    projects_dict = {}
    i = 0
    for project in projects:
        app = {
            f"project{i}": {
                "name": project.project_name,
                "description": project.project_description,
            }
        }
        projects_dict = {**projects_dict, **app}
        i += 1
    return projects_dict


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
            render_template(
                "profile.html", user=current_user, projects=get_all_user_projects()
            ),
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
