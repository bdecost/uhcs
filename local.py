# -*- coding: utf-8 -*-
"""
    mfeat.local
    ~~~~~~~

    This module provides a wrapper for vlfeat local image feature extraction

    :license: MIT, see LICENSE for more details.
"""
import numpy as np
from oct2py import octave

def random_sample(descriptors, fraction=0.1):
    """ randomly filter descriptors to thin out the training set for dictionary extraction """
    n = descriptors.shape[0]
    fraction = np.clip(fraction, 0.0, 1.0)
    n_keep = int(round(n * fraction))
    selection = np.random.choice(range(n), size=n_keep, replace=False)
    return descriptors[np.sort(selection)]

def sparse_sift(image, fraction=1.0):
    """ sparse oriented SIFT at Harris-LaPlace and Difference of Gaussians keypoints 
        use VLFEAT vl_covdet through octave; expects a grayscale image
    """
    octave.push("im", image)
    octave.eval("im = single(im);")
    octave.eval(
        "[kp,sift_hl] = vl_covdet(im, 'method', 'HarrisLaplace', 'EstimateOrientation', true); ")
    octave.eval(
        "[kp,sift_dog] = vl_covdet(im, 'method', 'DoG', 'EstimateOrientation', true); ")
    octave.eval("descrs = [sift_hl, sift_dog];")
    descriptors = octave.pull("descrs")

    # flip from column-major to row-major
    descriptors = descriptors.T

    if fraction < 1.0:
        descriptors = random_sample(descriptors, fraction)

    return descriptors

def dense_sift(image, fraction=1.0):
    """ dense SIFT
        use VLFEAT vl_phow through octave; expects a grayscale image
    """
    octave.push("im", image)
    octave.eval("im = single(im);")
    octave.eval("[kp,siftd] = vl_phow(im); ")
    descriptors = octave.pull("siftd")

    # flip from column-major to row-major
    descriptors = descriptors.T

    if fraction < 1.0:
        descriptors = random_sample(descriptors, fraction)
        
    return descriptors

