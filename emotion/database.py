import math
from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


__all__ = 'GRAM_LENGTH', 'Gram', 'Session'


Base = declarative_base()
Session = sessionmaker()
GRAM_LENGTH = 2


def _sigmoid(x):
    return math.tanh(x)


def _dsigmoid(y):
    return math.atanh(y)


class Gram(Base):
    NEIGHBOR_STRENGTH = 0.3
    __tablename__ = 'grams'

    id = Column(Integer, primary_key=True)
    gram = Column(String(GRAM_LENGTH), unique=True, nullable=False)

    anger = Column(Float, nullable=False, default=.0)
    interest = Column(Float, nullable=False, default=.0)
    joy = Column(Float, nullable=False, default=.0)
    trust = Column(Float, nullable=False, default=.0)

    def __init__(self, gram):
        self.gram = gram
        self.count = 0
        self.anger = 0
        self.interest = 0
        self.joy = 0
        self.trust = 0

    def apply_emotion(self, anger, interest, joy, trust):
        delta_anger = anger + self.NEIGHBOR_STRENGTH * (interest - trust)
        delta_interest = interest + self.NEIGHBOR_STRENGTH * (anger + joy)
        delta_joy = interest + self.NEIGHBOR_STRENGTH * (interest + trust)
        delta_trust = trust + self.NEIGHBOR_STRENGTH * (joy - anger)

        self.anger = _sigmoid(_dsigmoid(self.anger) + delta_anger)
        self.interest = _sigmoid(_dsigmoid(self.interest) + delta_interest)
        self.joy = _sigmoid(_dsigmoid(self.joy) + delta_joy)
        self.trust = _sigmoid(_dsigmoid(self.trust) + delta_trust)

    def __repr__(self):
        return '<{0} {1}>'.format(self.__class__.__name__, self.gram)
