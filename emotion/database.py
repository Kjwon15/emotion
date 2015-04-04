import math
from sqlalchemy import Column, Float, ForeignKey, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker


__all__ = 'GRAM_LENGTH', 'Base', 'Document', 'Gram', 'Session'


Base = declarative_base()
Session = sessionmaker()
GRAM_LENGTH = 2


association_table = Table(
    'association', Base.metadata,
    Column('document_id', String, ForeignKey('documents.text')),
    Column('gram_id', String, ForeignKey('grams.gram')))


class Document(Base):
    __tablename__ = 'documents'
    text = Column(String, primary_key=True, nullable=False)
    grams = relationship(
        'Gram', secondary=association_table, backref='documents')

    def __init__(self, text):
        self.text = text


class Gram(Base):
    __tablename__ = 'grams'

    gram = Column(String(GRAM_LENGTH), primary_key=True, nullable=False)

    anger = Column(Float, nullable=False, default=.0)
    interest = Column(Float, nullable=False, default=.0)
    joy = Column(Float, nullable=False, default=.0)
    trust = Column(Float, nullable=False, default=.0)
    idf = Column(Float, nullable=False)

    def __init__(self, gram):
        self.gram = gram
        self.count = 0
        self.anger = 0
        self.interest = 0
        self.joy = 0
        self.trust = 0

    def update_idf(self):
        session = Session()
        all_count = session.query(Document).count()
        document_count = len(self.documents)

        self.idf = math.log((all_count / (1.0 + document_count)) + 1)
        session.commit()

    def __repr__(self):
        return '<{0} {1}>'.format(self.__class__.__name__, self.gram)
