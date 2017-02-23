#!/usr/bin/env python

import re
import os
import h5py
import json
import click
import warnings
import numpy as np

import skimage
import skimage.io
import skimage.transform
import skimage.color

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

sns.set(style='white', context='poster')

import models
from models import Base, User, Collection, Sample, Micrograph, dbpath
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///data/microstructures.sqlite')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
db = DBSession()

def load_features(featuresfile, perplexity=40):
    keys, X = [], []

    with h5py.File(featuresfile, 'r') as f:
        g = f['perplexity-{}'.format(perplexity)]
        for key in g.keys():
            xx = g[key][...]
            if np.any(np.isnan(xx)):
                print('{} has NaN values in {}'.format(key, featuresfile))
                continue
            keys.append(key)
            X.append(g[key][...])

    X = np.array(X)
    return keys, X

def colorpad(image, color=np.array([1,1,0]), width=3):
    """ pad out a thumbnail with a colored border """
    i = skimage.color.gray2rgb(image)
    z = np.pad(i, ((3,3),(3,3), (0,0)), mode='constant', constant_values=(-1,-1))
    z[z[:,:,0] == -1 ] = color
    return z

def image_montage(X, images, bordercolors, mapsize=8192, thumbsize=256, bordersize=4, verbose=False):
    """ make image maps in an embedding space """

    # convert embedding coordinates to range (0,1)
    xmap = (1 + (X / (1.1*np.max(np.abs(X), axis=0)))) / 2
    
    map_shape = np.array([mapsize,mapsize,3])
    imagemap = np.ones(map_shape)

    # rescale max distance from origin to 1
    scale = np.max(np.abs(X[:,0:2]))

    for ids, image in enumerate(images):

        # get image position and border color
        pos = xmap[ids][:2]
        bordercolor = bordercolors[ids]
        
        # load image
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            im = skimage.io.imread(image, as_grey=True)

        # crop arbitrarily to square aspect ratio
        mindim = min(im.shape)
        cropped = im[:mindim,:mindim]
    
        # make thumbnail
        thumbnail = skimage.transform.resize(cropped, (thumbsize,thumbsize), order=1)        
        thumbnail = colorpad(thumbnail, color=bordercolor)

        thumb_width, thumb_height, _ = np.array(thumbnail.shape)
        y, x = np.round(pos * mapsize).astype(int)
        x = mapsize - x # image convention -- match scatter plot
        
        # place thumbnail into image map
        if verbose:
            print(thumbnail.shape)
            print('({},{})'.format(x,y))

        xstart = x-int(thumb_width/2)
        ystart = y-int(thumb_height/2)
    
        imagemap[xstart:xstart+thumb_width,ystart:ystart+thumb_height,:] = thumbnail

    return imagemap

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('featuresfile', nargs=1, type=click.Path())
@click.option('--mapsize', '-s', default=13, help='draw map of size 2**MAPSIZE')
@click.option('--thumbsize', '-t', default=256)
@click.option('--bordersize', '-b', default=4)
@click.option('--perplexity', '-p', default=40, help='t-SNE perplexity to use')
def tsne_map(featuresfile, mapsize, thumbsize, bordersize, perplexity):
    mapsize = 2**mapsize
    print('ok...')
    # expect root data directory to be in the working directory...
    path_list = os.path.normpath(featuresfile).split(os.sep)
    dataset_dir = os.path.join(*list(path_list[:-2]))
    print(featuresfile)
    
    engine = create_engine('sqlite:///data/microstructures.sqlite')
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    db = DBSession()

    keys, X = load_features(featuresfile, perplexity=perplexity)

    print(X.shape)
    
    labels, paths = [], []
    for key in keys:
        if 'crop' in featuresfile:
            key = key.split('-')[0]
        m = db.query(Micrograph).filter(Micrograph.micrograph_id == int(key)).one()
        labels.append(m.primary_microconstituent)

        basename, ext = os.path.splitext(m.path)
        paths.append('data/micrographs/micrograph{}{}'.format(m.micrograph_id, ext))

    # set thumbnail border colors -- keep consistent with scatter plots
    colornames = ["blue", "cerulean", "red", "dusty purple", "saffron", "dandelion", "green"]
    pal = sns.xkcd_palette(colornames)
    unique_labels = np.array(['spheroidite', 'spheroidite+widmanstatten', 'martensite', 'network',
                              'pearlite', 'pearlite+spheroidite', 'pearlite+widmanstatten'])

    cmap = {label: np.array(pal[idx]) for idx,label in enumerate(unique_labels)}
    colors = [cmap[label] for label in unique_labels]
    bordercolors = [cmap[label] for label in labels]
    

    # swap X[:,0] and X[:,1] to be consistent with how scatter plot displays images....
    XX = np.zeros_like(X)
    XX[:,0] = X[:,1]
    XX[:,1] = X[:,0]

    print('building image map (size = {})'.format(mapsize))
    imagemap = image_montage(XX, paths, bordercolors, mapsize=mapsize, thumbsize=thumbsize, bordersize=bordersize)

    basename = os.path.basename(featuresfile)
    basename, ext = os.path.splitext(basename)
    skimage.io.imsave('{}/figures/tsne/map/{}-bordered-tsne-map.png'.format(dataset_dir, basename), imagemap)

if __name__ == '__main__':
    tsne_map()
