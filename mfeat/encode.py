# -*- coding: utf-8 -*-
"""
    mfeat.encode
    ~~~~~~~

    This module provides implementations for image feature encoding methods,
    i.e. bag of words, VLAD, etc

    :license: MIT, see LICENSE for more details.
"""

import numpy as np
from scipy.spatial.distance import cdist

def assign_hard_labels(features, dictionary):
    # np.cdist computes pairwise distances between feature vectors and dictionary entries
    # use this with np.argmin to exhaustively search for the nearest cluster center
    # along axis 1 to vectorize over each SIFT vector
    labels = np.argmin(cdist(features, dictionary.cluster_centers_), axis=1)
    return labels

def bag_of_words(features, dictionary):
    """ construct bag of words representation given a feature array and a dictionary
        dictionary should be a sklearn.cluster.KMeans-like object 
    """
    labels = assign_hard_labels(features, dictionary)
    bow, __ = np.histogram(labels, bins=dictionary.n_clusters, density=True)

    return bow

def vlad(features, dictionary):
    """ construct VLAD features given an image and a dictionary 
        dictionary should be a scikit-learn clustering model (such as KMeans) """

    labels = assign_hard_labels(features, dictionary)

    # VLAD-encode features...
    # instead of counting feature occurrences, VLAD sums up the residuals within each cluster
    # use intra-normalization scheme from Arandjelovic 2013 -- All About VLAD
    residuals = []
    for label in range(dictionary.n_clusters):
        center = dictionary.cluster_centers_[label]
        X = features[labels == label]
        
        if X.size == 0:
            # no assignments to this visual word
            residuals.append(np.zeros(center.shape))
            continue
        
        residual = np.sum(X - center, axis=0)
        # L2-normalize each block of the VLAD -- i.e. each residual
        residual = residual / np.linalg.norm(residual)
        residuals.append(residual)

    # L2-normalize the final VLAD vector too
    X_vlad = np.concatenate(residuals)
    return X_vlad / np.linalg.norm(X_vlad)

def fast_fisher_encode(image_path, dictionary, extract_features='sparse'):
    """ construct Fisher Vector features given an image and a dictionary 
        dictionary should be a scikit-learn gaussian mixture model 
        fast version: like VLAD, only count the strongest component """
    raise NotImplementedError

    if extract_features == 'sparse':
        features_path = 'Fisher'
        features = sparse_sift(image_path)
    elif extract_features == 'dense':
        features_path = 'dFisher'
        features = dense_sift(image_path)

    posterior_prob = dictionary.predict_proba(features)
    N = features.shape[0]

    mean_deviation, covar_deviation = [], []
    for mode in range(dictionary.n_components):
        q = posterior_prob[:,mode][:,newaxis] # column vector....
        mean = dictionary.means_[mode]
        covar = dictionary.covars_[mode]
    
        intermediate_dev = (features - mean) / covar

        mean_dev = (1/(N*sqrt(pi))) * sum(q * intermediate_dev, axis=0)

        covar_dev = (1/(N*sqrt(2*pi))) * sum(q * (square(intermediate_dev) - 1), axis=0)

        # L2-normalize each block
        mean_deviation.append(mean_dev / norm(mean_dev))
        covar_deviation.append(covar_dev / norm(covar_dev))
    
    mean_deviation = concatenate(mean_deviation)
    covar_deviation = concatenate(covar_deviation)
    fv = concatenate((mean_deviation, covar_deviation))

    # L2 normalize the final Fisher vector too
    return fv / norm(fv)
