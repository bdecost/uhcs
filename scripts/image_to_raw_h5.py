import os
import h5py
import json
import click
import numpy as np
from skimage import io

def load_image(image_path, barheight=38):
    # crop scale bars off Matt Hecht's SEM images

    image = io.imread(image_path, as_grey=True)
    if barheight > 0:
        image = image[:-barheight,:]
    return image

@click.command()
@click.argument('micrographs_json', nargs=1, type=click.Path())
def image_to_raw_h5(micrographs_json):
    parent, __ = os.path.split(micrographs_json)
    featurefile = os.path.join(parent, 'features', 'raw.h5')

    barheight = 0
    if 'full' in micrographs_json:
        barheight = 38
        
    print(featurefile)

    # obtain a dataset
    with open(micrographs_json, 'r') as f:
        micrograph_dataset = json.load(f)

    # work with sorted micrograph keys...
    keys = sorted(micrograph_dataset.keys())
    micrographs = [micrograph_dataset[key] for key in keys]
    # micrographs = [io.load_image(m) for m in micrographs]

    
    with h5py.File(featurefile, 'w') as f:
        for key in keys:
            m = micrograph_dataset[key]
            micrograph = load_image(m, barheight=barheight)
            f[key] = micrograph.flatten()


if __name__ == '__main__':
    image_to_raw_h5()
