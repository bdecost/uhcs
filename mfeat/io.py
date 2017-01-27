# -*- coding: utf-8 -*-
"""
    mfeat.io
    ~~~~~~~

    This module provides simple i/o routines for micrograph datasets

    :license: MIT, see LICENSE for more details.
"""

from skimage import io

def load_image(image_path, barheight=0):
    # crop scale bars off Matt Hecht's SEM images
    # UHCS images have barheight=38 px
    image = io.imread(image_path, as_grey=True)
    if barheight > 0:
        image = image[:-barheight,:]
    return image
