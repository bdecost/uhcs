import os
import glob
import json
import fnmatch
import itertools

# look for all micrographs, which have extensions ['.tif', '.png', '.jpg']
extensions = ['*.tif', '*.png', '*.jpg', '*.JPG', '*.bmp']

images = []
toplevel_dir = './data/micrograph_dump'
images_jsonfile = 'data/micrographs.json'

for dirpath, dirnames, filenames in os.walk(toplevel_dir):

    # construct relative file paths
    filepaths = [os.path.join(dirpath, filename) for filename in filenames]
        
    # filter out images by matching file extensions
    ims = (fnmatch.filter(filepaths, extension) for extension in extensions)
    ims = list(itertools.chain.from_iterable(ims))
    images = images + ims

image_paths = {key: image for key,image in enumerate(images)}
with open(images_jsonfile, 'w') as f:
    json.dump(image_paths, f)
