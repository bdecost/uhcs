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

def tensor_to_features(X, subsample=None):
    """ convert feature map tensor to numpy data matrix {nsamples, nchannels} """
    
    # transpose array so that map dimensions are on the last axis
    features = X.transpose(0,2,3,1) # to [batch, height, width, channels]
    features = features.reshape((-1, features.shape[-1])) # to [feature, channels]

    if subsample >= 1.0 or sumsample <= 0:
        subsample = None

    if subsample is not None:
        choice = np.sort(
            np.random.choice(range(features.shape[0]), size=subsample, replace=False)
        )
        features = features[choice]
        
    return features

def cnn_features(image, layername, fraction=None):
    """ use keras to obtain cnn feature map """
    # TODO: refactor calling code to directly call keras model
    
    model = Model(input=cnn.input, output=cnn.get_layer(layername).output)
    out = model.predict(image_tensor(image))
        
    return tensor_to_features(out, subsample=fraction)

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

