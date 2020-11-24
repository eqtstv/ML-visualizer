from urllib.parse import urljoin, urlparse

from flask import request
from flask_login import UserMixin
from flask_wtf import FlaskForm
from ml_visualizer.database import Base
from sqlalchemy import Column, Float, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import relationship
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


class Projects(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("Users.id"))
    project_name = Column(String(64), unique=True)
    project_description = Column(String(64))

    def __init__(self, user_id=None, project_name=None, project_description=None):
        self.user_id = user_id
        self.project_name = project_name
        self.project_description = project_description

    def __repr__(self):
        return "<Project name>" % (self.project_name)


class LogTraining(Base):
    __tablename__ = "log_training"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("Users.id"))
    project_name = Column(String(64), ForeignKey("projects.project_name"))
    step = Column(Integer)
    batch = Column(Integer)
    train_accuracy = Column(Float(50))
    train_loss = Column(Float(50))

    def __init__(
        self,
        user_id=None,
        project_name=None,
        step=None,
        batch=None,
        train_accuracy=None,
        train_loss=None,
    ):
        self.user_id = user_id
        self.project_name = project_name
        self.step = step
        self.batch = batch
        self.train_accuracy = train_accuracy
        self.train_loss = train_loss

    def __repr__(self):
        return "<Batch %r>" % (self.batch)


class LogValidation(Base):
    __tablename__ = "log_validation"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("Users.id"))
    project_name = Column(String(64), ForeignKey("projects.project_name"))
    step = Column(Integer)
    val_accuracy = Column(Float(50))
    val_loss = Column(Float(50))
    epoch = Column(Integer)
    epoch_time = Column(Float(50))

    def __init__(
        self,
        user_id=None,
        project_name=None,
        step=None,
        val_accuracy=None,
        val_loss=None,
        epoch=None,
        epoch_time=None,
    ):
        self.user_id = user_id
        self.project_name = project_name
        self.step = step
        self.val_accuracy = val_accuracy
        self.val_loss = val_loss
        self.epoch = epoch
        self.epoch_time = epoch_time

    def __repr__(self):
        return "<Epoch %r>" % (self.epoch)


class ModelParameters(Base):
    __tablename__ = "model_params"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("Users.id"))
    project_name = Column(String(64), ForeignKey("projects.project_name"))
    tracking_precision = Column(Float(50))
    no_steps = Column(Integer)
    epochs = Column(Integer)
    batch_split = Column(Integer)
    max_batch_step = Column(Integer)
    steps_in_batch = Column(Integer)
    no_tracked_steps = Column(Integer)
    total_params = Column(Integer)

    def __init__(
        self,
        user_id=None,
        project_name=None,
        tracking_precision=None,
        no_steps=None,
        epochs=None,
        batch_split=None,
        max_batch_step=None,
        steps_in_batch=None,
        no_tracked_steps=None,
        total_params=None,
    ):
        self.user_id = user_id
        self.project_name = project_name
        self.tracking_precision = tracking_precision
        self.no_steps = no_steps
        self.epochs = epochs
        self.batch_split = batch_split
        self.max_batch_step = max_batch_step
        self.steps_in_batch = steps_in_batch
        self.no_tracked_steps = no_tracked_steps
        self.total_params = total_params


class ModelSummaryDB(Base):
    __tablename__ = "model_summary"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("Users.id"))
    project_name = Column(String(64), ForeignKey("projects.project_name"))
    class_name = Column(String(64))
    config = Column(JSON)

    def __init__(self, user_id=None, project_name=None, class_name=None, config=None):
        self.user_id = user_id
        self.project_name = project_name
        self.class_name = class_name
        self.config = config


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    remember_me = BooleanField("Remember Me")
    submit = SubmitField("Sign In")


def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ("http", "https") and ref_url.netloc == test_url.netloc
