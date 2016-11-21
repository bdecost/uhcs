# -*- coding: utf-8 -*-
"""
    mfeat.cnn
    ~~~~~~~

    This module wraps the keras convnet implementations

    :license: MIT, see LICENSE for more details.
"""

import numpy as np
from scipy.misc import imresize
from skimage.color import gray2rgb

from keras.models import Model
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input

cnn = VGG16(include_top=False, weights='imagenet')
layer_id = {layer.name: idx for idx, layer in enumerate(cnn.layers)}

def image_tensor(image):
    """ replicate a grayscale image onto the three channels of an RGB image
        and reshape into a tensor appropriate for keras
    """
    image3d = gray2rgb(image).astype(np.float32)
    x = 255*image3d.transpose((2,0,1))
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def cnn_features(image, layername, fraction=1.0):
    """ use keras to obtain cnn feature map """
    output_layer = [cnn.layers[layer_id[layername]].output]
    cnn_map = K.function([cnn.layers[0].input], output_layer)
    
    t = image_tensor(image)
    out = cnn_map([t])[0]

    n_channels = out.shape[1]
    n_points = out.shape[2] * out.shape[3]

    # transpose array so that map dimensions are on the last axis
    # reshape to a standard data matrix (samples, channels)
    features = out.transpose(2,3,1,0)
    features = features.reshape((n_points, n_channels))

    if fraction < 1.0:
        # randomly subsample some feature map pixels for dictionary learning
        sample = np.sort(
            np.random.choice(range(n_points), int(fraction*n_points), replace=False)
        )
        features = features[sample]
    
    return features

def multiscale_cnn_features(image, layername, fraction=1.0,
                            scale=1/np.sqrt(2), n_downsample=3, n_upsample=1):
    """ pool feature maps from scale-space pyramid to increase scale-invariance """
    resample_exponents = range(min(-n_upsample, 0), max(0, n_downsample))

    f = []
    for exponent in resample_exponents:
        # resample image with bilinear interpolation
        im = imresize(image, scale**exponent)
        f.append(cnn_features(im, layername, fraction=fraction))

    return np.vstack(f)

