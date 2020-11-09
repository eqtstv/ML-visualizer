from sqlalchemy import Column, Integer, String, Float
from database.database import Base


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
