from urllib.parse import urljoin, urlparse

from flask import (
    abort,
    flash,
    make_response,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_restful import Resource
from werkzeug.security import check_password_hash, generate_password_hash

from ml_visualizer.database import Base, db_session
from ml_visualizer.models import User

login_manager = LoginManager()
login_manager.login_view = "notlogged"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)


def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ("http", "https") and ref_url.netloc == test_url.netloc


class Signup(Resource):
    def get(self):
        return make_response(
            render_template("signup.html", user=current_user),
            200,
            {"Content-Type": "text/html"},
        )

    def post(self):
        email = request.form.get("email")
        username = request.form.get("name")
        password = request.form.get("password")

        user = User.query.filter_by(
            email=email
        ).first()  # if this returns a user, then the email already exists in database

        if (
            user
        ):  # if a user is found, we want to redirect back to signup page so user can try again
            return make_response(
                redirect(url_for("signup")), 200, {"Content-Type": "text/html"}
            )

        # create a new user with the form data. Hash the password so the plaintext version isn't saved.
        new_user = User(
            email=email,
            username=username,
            password_hash=generate_password_hash(password, method="sha256"),
        )

        # add the new user to the database
        db_session.add(new_user)
        db_session.commit()

        return make_response(
            redirect(url_for("login")), 200, {"Content-Type": "text/html"}
        )

    def put(self):
        data = request.json
        email = data["email"]
        username = data["name"]
        password = data["password"]

        user = User.query.filter_by(
            email=email
        ).first()  # if this returns a user, then the email already exists in database

        if (
            user
        ):  # if a user is found, we want to redirect back to signup page so user can try again
            return make_response(
                redirect(url_for("signup")), 200, {"Content-Type": "text/html"}
            )

        # create a new user with the form data. Hash the password so the plaintext version isn't saved.
        new_user = User(
            email=email,
            username=username,
            password_hash=generate_password_hash(password, method="sha256"),
        )

        # add the new user to the database
        db_session.add(new_user)
        db_session.commit()

        return make_response(
            redirect(url_for("login")), 200, {"Content-Type": "text/html"}
        )


class Login(Resource):
    def get(self):
        return make_response(
            render_template("login.html", user=current_user),
            200,
            {"Content-Type": "text/html"},
        )

    def post(self):
        email = request.form.get("email")
        password = request.form.get("password")
        remember = True if request.form.get("remember") else False

        user = User.query.filter_by(email=email).first()

        # check if the user actually exists
        # take the user-supplied password, hash it, and compare it to the hashed password in the database
        if not user or not check_password_hash(user.password_hash, password):
            flash("Please check your login details and try again.")
            return redirect(
                url_for("login")
            )  # if the user doesn't exist or password is wrong, reload the page

        # if the above check passes, then we know the user has the right credentials
        next = request.args.get("next")
        if not is_safe_url(next):
            return abort(400)

        login_user(user)
        return redirect(url_for("profile"))


class Logout(Resource):
    @login_required
    def get(self):
        logout_user()
        return make_response(
            render_template("logout.html", user=current_user),
            200,
            {"Content-Type": "text/html"},
        )
