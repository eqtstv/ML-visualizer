from urllib.parse import urljoin, urlparse

from flask import request
from flask_login import UserMixin
from flask_wtf import FlaskForm
from ml_visualizer.database.database import Base
from sqlalchemy import Column, Integer, String, Float
from werkzeug.security import check_password_hash, generate_password_hash
from wtforms import BooleanField, PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired


class User(UserMixin, Base):
    __tablename__ = "Users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), index=True, unique=True)
    email = Column(String(64), index=True, unique=True)
    password_hash = Column(String(64))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return "<User{}>".format(self.username)


class LogTraining(Base):
    __tablename__ = "log_training"
    id = Column(Integer, primary_key=True)
    step = Column(Integer)
    batch = Column(Integer)
    train_accuracy = Column(Float(50))
    train_loss = Column(Float(50))

    def __init__(self, step=None, batch=None, train_accuracy=None, train_loss=None):
        self.step = step
        self.batch = batch
        self.train_accuracy = train_accuracy
        self.train_loss = train_loss

    def __repr__(self):
        return "<Batch %r>" % (self.batch)


class LogValidation(Base):
    __tablename__ = "log_validation"
    id = Column(Integer, primary_key=True)
    step = Column(Integer)
    val_accuracy = Column(Float(50))
    val_loss = Column(Float(50))
    epoch = Column(Integer)
    epoch_time = Column(Float(50))

    def __init__(
        self, step=None, val_accuracy=None, val_loss=None, epoch=None, epoch_time=None
    ):
        self.step = step
        self.val_accuracy = val_accuracy
        self.val_loss = val_loss
        self.epoch = epoch
        self.epoch_time = epoch_time

    def __repr__(self):
        return "<Epoch %r>" % (self.epoch)


class Projects(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    project_name = Column(String(64), unique=True)
    project_description = Column(String(64))

    def __init__(self, project_name=None, project_description=None):
        self.project_name = project_name
        self.project_description = project_description

    def __repr__(self):
        return "<Project name>" % (self.project_name)


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    remember_me = BooleanField("Remember Me")
    submit = SubmitField("Sign In")


def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ("http", "https") and ref_url.netloc == test_url.netloc
