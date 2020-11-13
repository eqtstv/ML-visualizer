import unittest

import sqlalchemy
from sqlalchemy import inspect, Column, Float, Integer
from ml_visualizer.database.database import Base, db_session, engine, init_db
from ml_visualizer.database.models import LogTraining, LogValidation


class TestDatabase(unittest.TestCase):
    def test_Base_type(self):
        # Base should be an instance of SQLAlchemy declarative database
        self.assertIsInstance(Base, sqlalchemy.ext.declarative.api.DeclarativeMeta)

    def test_db_session_type(self):
        # db_session should an be instance if sqlalchemy scoped_session
        self.assertIsInstance(db_session, sqlalchemy.orm.scoping.scoped_session)

    def test_engine_type(self):
        # engine should be an instance of sqlalchemy Engine
        self.assertIsInstance(engine, sqlalchemy.engine.base.Engine)


class TestLogTraining(unittest.TestCase):
    def test_log_training_table_name(self):
        self.assertEqual(LogTraining.__tablename__, "log_training")

    def test_log_training_columns(self):
        # GIVEN table columns
        columns = [(column.name, column.type) for column in inspect(LogTraining).c]

        # AND valid column names and types
        valid_column_names = ["id", "step", "batch", "train_accuracy", "train_loss"]
        valid_column_types = [Integer, Integer, Integer, Float, Float]

        # THEN colums should have valid names and types
        for i, k in enumerate(columns):
            self.assertEqual(columns[i][0], valid_column_names[i])
            self.assertIsInstance(columns[i][1], valid_column_types[i])


class TestLogValidation(unittest.TestCase):
    def test_log_validaton_table_name(self):
        self.assertEqual(LogValidation.__tablename__, "log_validation")

    def test_log_validation_columns(self):
        # GIVEN table columns
        columns = [(column.name, column.type) for column in inspect(LogValidation).c]

        # AND valid column names and types
        valid_column_names = [
            "id",
            "step",
            "val_accuracy",
            "val_loss",
            "epoch",
            "epoch_time",
        ]
        valid_column_types = [Integer, Integer, Float, Float, Integer, Float]

        # THEN colums should have valid names and types
        for i, k in enumerate(columns):
            self.assertEqual(columns[i][0], valid_column_names[i])
            self.assertIsInstance(columns[i][1], valid_column_types[i])
