""" Database models for microstructure dataset """

import os
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()

dbpath = 'sqlite:///data/microstructures.sqlite'

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username =  Column(String(250))
    firstname = Column(String(250))
    lastname =  Column(String(250))
    email =     Column(String(250))
    micrographs = relationship('Micrograph')
        
class Collection(Base):
    __tablename__ = 'collection'
    id =   Column(Integer, primary_key=True)
    name = Column(String(250))
    
class Sample(Base):
    __tablename__ = 'sample'
    id = Column(Integer, primary_key=True)
    label = Column(String(250))
    anneal_time = Column(Float)
    anneal_time_unit = Column(String(16))
    anneal_temperature = Column(Float)
    anneal_temp_unit = Column(String(16))
    cool = Column(String(16))
    micrographs = relationship('Micrograph')
    
class Micrograph(Base):
    __tablename__ = 'micrograph'
    id = Column(Integer, primary_key=True)
    path =             Column(String())
    contributor =      Column(Integer, ForeignKey('user.id'))
    micron_bar =       Column(Float)
    micron_bar_units = Column(String(64))
    micron_bar_px =    Column(Integer)
    magnification =    Column(Integer)
    detector =         Column(String(16))
    sample_id =        Column(Integer, ForeignKey('sample.id'))
    sample =           relationship('Sample', back_populates='micrographs')
    # user_id =          Column(Integer, ForeignKey('user.id'))
    user =             relationship('User', back_populates='micrographs')
    mstructure_class = Column(String(250))


if __name__ == '__main__':
    engine = create_engine(dbpath)

    Base.metadata.create_all(engine)
