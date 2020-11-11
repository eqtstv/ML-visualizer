from flask_restful import Resource
from ml_visualizer.database.database import Base, db_session


class ClearData(Resource):
    def delete(self):
        meta = Base.metadata
        for table in reversed(meta.sorted_tables):
            db_session.execute(table.delete())
        db_session.commit()
