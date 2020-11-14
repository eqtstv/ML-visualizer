import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

file_path = os.path.abspath(os.getcwd()) + "/users.db"

engine = create_engine(
    f"sqlite:///{file_path}",
    convert_unicode=True,
    connect_args={"check_same_thread": False},  # ???
)

db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)
users_db = declarative_base()
users_db.query = db_session.query_property()


def init_users_db():
    # import all modules here that might define models so that
    # they will be registered properly on the metadata.  Otherwise
    # you will have to import them first before calling init_db()
    from ml_visualizer.auth.database.models import User

    users_db.metadata.create_all(bind=engine)
