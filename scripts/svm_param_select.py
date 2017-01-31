#!/usr/bin/env python
import os
import h5py
import json
import click
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# needed with slurm to see local python library under working dir
import sys
sys.path.append(os.path.join(os.getcwd(), 'code'))

import models
from models import Base, User, Collection, Sample, Micrograph, dbpath
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///data/microstructures.sqlite')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
db = DBSession()

def load_representations(datafile):
    # grab image representations from hdf5 file
    keys, features = [], []

    with h5py.File(datafile, 'r') as f:
        for key in f:
            keys.append(key)
            features.append(f[key][...])

    return np.array(keys), np.array(features)

def select_balanced_dataset(labels, X, n_per_class=50, seed=0):
    """ select a balanced dataset for cross-validation """
    np.random.seed(seed) # set seed to enable deterministic training set
    selection = []
    tlabel = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        sel = np.random.choice(idx, n_per_class, replace=False)
        selection.append(sel)
        tlabel += [label]*n_per_class

    selection = np.concatenate(selection)
    l = np.array(tlabel)
    X = X[selection]
    np.random.seed() # reset seed to randomize cv folds across runs
    return l, X, selection

def cv_loop_chi2(labels, X, cv, C=1, n_repeats=1):
    tscore, vscore = [], []
    for repeat in range(n_repeats):
        for train, test in cv.split(X, labels):
            # follow Zhang et al (2007) in setting gamma
            gamma = -1 / np.mean(additive_chi2_kernel(X[train]))
            clf = SVC(kernel=chi2_kernel, gamma=gamma, C=C,
                      class_weight='balanced', decision_function_shape='ovr', cache_size=2048)
    
            clf.fit(X[train], labels[train])
            tscore.append(clf.score(X[train], labels[train]))
            vscore.append(clf.score(X[test], labels[test]))

    print('{} +/- {}'.format(np.mean(vscore), np.std(vscore, ddof=1)))
    return np.mean(vscore), np.std(vscore, ddof=1), np.mean(tscore), np.std(tscore, ddof=1)

def cv_loop_linear(labels, X, cv, C=1, n_repeats=1, reduce_dim=None):
    tscore, vscore = [], []
    clf = SVC(kernel='linear', C=C,
              class_weight='balanced', decision_function_shape='ovr', cache_size=2048)
    for repeat in range(n_repeats):
        for train, test in cv.split(X, labels):

            if reduce_dim:
                pca = PCA(n_components=reduce_dim).fit(X[train])
                Xtrain = pca.transform(X[train])
                Xtest = pca.transform(X[test])
            else:
                Xtrain = X[train]
                Xtest = X[test]

            # L2-normalize instances for linear SVM following Vedaldi and Zisserman
            # Efficient Additive Kernels via Explicit Feature Maps
            # Also: Perronnin, Sanchez, and Mensink
            # Improving the Fisher kernel for large-scale image classification
            Xtrain = Xtrain / np.linalg.norm(Xtrain, axis=1)[:,np.newaxis]
            Xtest = Xtest / np.linalg.norm(Xtest, axis=1)[:,np.newaxis]
    
            clf.fit(Xtrain, labels[train])
            tscore.append(clf.score(Xtrain, labels[train]))
            vscore.append(clf.score(Xtest, labels[test]))

    print('{} +/- {}'.format(np.mean(vscore), np.std(vscore, ddof=1)))
    return np.mean(vscore), np.std(vscore, ddof=1), np.mean(tscore), np.std(tscore, ddof=1)

@click.command()
@click.argument('datafile', type=click.Path())
@click.option('--kernel', '-k', type=click.Choice(['linear', 'chi2']), default='linear')
@click.option('--margin-param', '-C', type=float, default=None)
@click.option('--n-per-class', '-n', type=int, default=50)
@click.option('--n-repeats', '-r', type=int, default=1)
@click.option('--reduce-dim', '-d', type=int, default=None)
@click.option('--seed', '-s', type=int, default=None)
def svm_param_select(datafile, kernel, margin_param, n_per_class, n_repeats, reduce_dim, seed):
    # datafile = './data/full/features/vgg16_block5_conv3-vlad-32.h5'

    # if margin_param is specified, record the results
    if margin_param is not None:
        resultsdir = 'svmresults'
    else:
        resultsdir = 'svm'
    resultsfile = datafile.replace('features', resultsdir).replace('h5', 'json')
    
    try:
        os.makedirs(os.path.dirname(resultsfile))
    except FileExistsError:
        pass

    keys, features = load_representations(datafile)

    labels = []
    for key in keys:
        if '-' in key:
            # deal with cropped micrographs: key -> Micrograph.id-UL
            m_id, quadrant = key.split('-')
        else:
            m_id = key
        m = db.query(Micrograph).filter(Micrograph.micrograph_id == int(m_id)).one()
        labels.append(m.primary_microconstituent)
    labels = np.array(labels)

    # simplify: get primary microconstituent; throw out martensite
    primary_label = np.array([label.split('+')[0] for label in labels])
    k = np.array(keys)[primary_label != 'martensite']
    l = primary_label[primary_label != 'martensite']
    X = features[primary_label != 'martensite']

    l, X, sel = select_balanced_dataset(l, X, n_per_class=n_per_class, seed=seed)

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    # cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1)

    print(datafile)
    results = {
        'training_set': list(keys[sel]),
        'kernel': kernel,
        'n_per_class': n_per_class,
        'seed': seed,
        'n_repeats': n_repeats,
        'cv_C': {}
    }
    

    if margin_param is not None:
        C_range = [margin_param]
    else:
        C_range = np.logspace(-12, 2, 15, base=2)
        
    for C in C_range:
        print(C)
        if kernel == 'linear':
            score, std, tscore, tstd = cv_loop_linear(l, X, cv, C=C,
                                                      n_repeats=n_repeats, reduce_dim=reduce_dim)
        elif kernel == 'chi2':
            score, std, tscore, tstd = cv_loop_chi2(l, X, cv, C=C, n_repeats=n_repeats)
            
        results['cv_C'][C] = {
            'score': score,
            'std': std,
            'tscore': tscore,
            'tstd': tstd
        }
        
    with open(resultsfile, 'w') as jf:
        json.dump(results, jf)


if __name__ == '__main__':
    svm_param_select()
