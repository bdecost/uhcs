#!/usr/bin/env python

import os
import json
import numpy as np

import models
from models import Base, User, Collection, Sample, Micrograph, dbpath
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///data/microstructures.sqlite')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
db = DBSession()

c = ['pearlite+widmanstatten', 'spheroidite', 'spheroidite+widmanstatten',
    'martensite', 'network', 'pearlite', 'pearlite+spheroidite']
micrographs = db.query(Micrograph).filter(Micrograph.mstructure_class.in_(c)).all()

full = {}
cropped = {}
for micrograph in micrographs:
    # print('{}: {}'.format(micrograph.mstructure_class, micrograph.path))
    prefix, ext = os.path.splitext(micrograph.path)
    path = 'data/micrographs/micrograph{}{}'.format(micrograph.id, ext)
    print('{}: {}'.format(micrograph.mstructure_class, path))
    full[micrograph.id] = path
    
    for crop in ('UL', 'UR', 'LL', 'LR'):
        path = 'data/crops/micrograph{}-crop{}.tif'.format(micrograph.id, crop)
        key = '{}-crop{}'.format(micrograph.id, crop)
        cropped[key] = path



with open('data/full/micrographs.json', 'w') as f:
    json.dump(full, f)

with open('data/cropped/micrographs.json', 'w') as f:
    json.dump(cropped, f)
