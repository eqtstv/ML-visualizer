from flask_jwt_extended import jwt_required
from flask_restful import Resource
from ml_visualizer.database.database import Base, db_session


class ClearData(Resource):
    @jwt_required
    def delete(self):
        meta = Base.metadata
        for table in reversed(meta.sorted_tables):
            db_session.execute(table.delete())
        db_session.commit()
