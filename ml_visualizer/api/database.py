import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

file_path = os.path.abspath(os.getcwd()) + "\logs.db"

engine = create_engine(f"sqlite:///{file_path}", convert_unicode=True)
db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)
Base = declarative_base()
Base.query = db_session.query_property()


def init_db():
    # import all modules here that might define models so that
    # they will be registered properly on the metadata.  Otherwise
    # you will have to import them first before calling init_db()
    from ml_visualizer.api.models import LogTraining, LogValidation

    Base.metadata.create_all(bind=engine)
