# uhcs

code accompanying 'Exploring the microstructure manifold: image texture representations applied to ultrahigh carbon steel microstructures' -- Brian DeCost, Toby Francis, and Elizabeth Holm. [arxiv.org](http://arxiv.org/abs/1702.01117)

This paper compares SIFT-based image representations with convolutional neural network representations for complex microstructures found in ultrahigh carbon steels.
The micrographs were collected by Matt Hecht at CMU through two studies: [a network morphology study](https://scholar.google.com/scholar?oi=bibs&cluster=16995291491472547776&btnI=1&hl=en) and a [spheroidite morphology study](https://dx.doi.org/10.1007/s11661-017-4012-2).
The UHCS microstructure  dataset is available on [materialsdata.nist.gov](https://materialsdata.nist.gov) under a Creative Commons license at [https://hdl.handle.net/11256/940](https://hdl.handle.net/11256/940), and will be documented by an IMMI manuscript (submitted 14 April 2017).
Please cite use of the UHCS microstructure data as
```TeX
@misc{uhcsdata,
  title={Ultrahigh Carbon Steel Micrographs},
  author = {Hecht, Matthew D. and DeCost, Brian L. and Francis, Toby and Holm, Elizabeth A. and Picard, Yoosuf N. and Webler, Bryan A.},
  howpublished={\url{https://hdl.handle.net/11256/940}}
}
```

## workflow
The `mfeat` module contains feature extraction code wrapping vlfeat (for SIFT features) and keras (for convnet features).
`scripts` contains code which applies this module to the uhcs micrograph dataset, perform the classification experiments from the manuscript, apply some dimensionality reduction algorithms, and create the thumbnail visualizations. Generally, each stage in the pipeline consists of a python script along with a shell script that launches a slurm task for each image representation we looked at.

1: unpack dataset (`microstructures.sqlite`, `micrographs/*.{tif,png,jpg}`, etc)
```sh
# bash download.sh
# get data from NIST for this project
# http://hdl.handle.net/11256/940
NIST_DATASET=11256/940
NIST_DATASET_URL=https://materialsdata.nist.gov/dspace/xmlui/bitstream/handle/${NIST_DATASET}

DATADIR=data

echo "download data files into DATADIR=${DATADIR}"

# download metadata
curl ${NIST_DATASET_URL}/microstructures.sqlite -o ${DATADIR}/microstructures.sqlite

# download micrographs
curl ${NIST_DATASET_URL}/micrographs.zip -o ${DATADIR}/micrographs.zip
unzip ${DATADIR}/micrographs.zip -d ${DATADIR}

```

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
