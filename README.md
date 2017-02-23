# uhcs

code accompanying 'Exploring the microstructure manifold: image texture representations applied to ultrahigh carbon steel microstructures' -- Brian DeCost, Toby Francis, and Elizabeth Holm. [arxiv.org](http://arxiv.org/abs/1702.01117)

This paper compares SIFT-based image representations with convolutional neural network representations for complex microstructures found in ultrahigh carbon steels.
The micrographs were collected by Matt Hecht at CMU through two studies: [a network morphology study](https://scholar.google.com/scholar?oi=bibs&cluster=16995291491472547776&btnI=1&hl=en) and a spheroidite morphology study (accepted for publication in Met Trans A).
We plan to make the data available soon in the NIST repository (in conjunction with an IMMI manuscript submission).

## workflow
The `mfeat` module contains feature extraction code wrapping vlfeat (for SIFT features) and keras (for convnet features).
`scripts` contains code which applies this module to the uhcs micrograph dataset, perform the classification experiments from the manuscript, apply some dimensionality reduction algorithms, and create the thumbnail visualizations. Generally, each stage in the pipeline consists of a python script along with a shell script that launches a slurm task for each image representation we looked at.

1: unpack dataset (`microstructures.sqlite`, `micrographs/*.{tif,png,jpg}`, etc)

2: pre-crop all the micrographs:
```sh
python scripts/crop_micrographs.py
```
3: generate json files mapping image keys to file paths (for inertia reasons...)
```sh
python scripts/enumerate_dataset.py
```

4: compute microstructure representations.
Representations are stored in hdf5 at `data/${format}/features/${representation}.h5`
```sh
bash scripts/compute_representations.sh
```

5: Run svm experiments. Cross-validation results stored in `data/${format}/svm/${representation}.json`
```sh
# primary microconstituent:
bash scripts/svm_result.sh

# annealing condition:
bash scripts/sample_svm.sh
```

6: run t-SNE (or other dimensionality reduction method...). Results stored in `data/${format}/tsne/${representation}.h5`
```sh
bash scripts/tsne_embed.sh
# bash scripts/manifold_embed.sh
```

7: make t-SNE thumbnail maps. Thumbnail maps are stored at `data/${format}/tsne/${representation}.png`
```sh
bash scripts/tsne_map.sh
```
