import unittest

import sqlalchemy
from ml_visualizer.database import Base, db_session, engine
from ml_visualizer.models import LogTraining, LogValidation, Projects
from sqlalchemy import Float, Integer, String, inspect


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
        valid_column_names = [
            "id",
            "user_id",
            "project_name",
            "step",
            "batch",
            "train_accuracy",
            "train_loss",
        ]
        valid_column_types = [Integer, Integer, String, Integer, Integer, Float, Float]

        # THEN colums should have valid names and types
        for i, k in enumerate(columns):
            self.assertEqual(columns[i][0], valid_column_names[i])
            self.assertIsInstance(columns[i][1], valid_column_types[i])

    def test_log_training_init(self):
        user_id = 1
        project_name = "my_project"
        step = 1
        batch = 10
        train_acc = 0.7
        train_loss = 0.5

        init_row = LogTraining(
            user_id, project_name, step, batch, train_acc, train_loss
        )
        self.assertEqual(init_row.user_id, user_id)
        self.assertEqual(init_row.project_name, project_name)
        self.assertEqual(init_row.step, step)
        self.assertEqual(init_row.batch, batch)
        self.assertEqual(init_row.train_accuracy, train_acc)
        self.assertEqual(init_row.train_loss, train_loss)


class TestLogValidation(unittest.TestCase):
    def test_log_validaton_table_name(self):
        self.assertEqual(LogValidation.__tablename__, "log_validation")

    def test_log_validation_columns(self):
        # GIVEN table columns
        columns = [(column.name, column.type) for column in inspect(LogValidation).c]

        # AND valid column names and types
        valid_column_names = [
            "id",
            "user_id",
            "project_name",
            "step",
            "val_accuracy",
            "val_loss",
            "epoch",
            "epoch_time",
        ]
        valid_column_types = [
            Integer,
            Integer,
            String,
            Integer,
            Float,
            Float,
            Integer,
            Float,
        ]

        # THEN colums should have valid names and types
        for i, k in enumerate(columns):
            self.assertEqual(columns[i][0], valid_column_names[i])
            self.assertIsInstance(columns[i][1], valid_column_types[i])

    def test_log_validation_init(self):
        user_id = 1
        project_name = "my_project"
        step = 10
        val_acc = 0.6
        val_loss = 0.7
        epoch = 3
        epoch_time = 4.2

        init_row = LogValidation(
            user_id, project_name, step, val_acc, val_loss, epoch, epoch_time
        )
        self.assertEqual(init_row.user_id, user_id)
        self.assertEqual(init_row.project_name, project_name)
        self.assertEqual(init_row.step, step)
        self.assertEqual(init_row.val_accuracy, val_acc)
        self.assertEqual(init_row.val_loss, val_loss)
        self.assertEqual(init_row.epoch, epoch)
        self.assertEqual(init_row.epoch_time, epoch_time)


class TestProjects(unittest.TestCase):
    def test_init_projects(self):
        user_id = 1
        project_name = "myproject"
        project_description = "my project description"

        init_row = Projects(user_id, project_name, project_description)
        self.assertEqual(init_row.user_id, user_id)
        self.assertEqual(init_row.project_name, project_name)
        self.assertEqual(init_row.project_description, project_description)
