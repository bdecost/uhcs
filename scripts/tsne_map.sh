#!/bin/bash

DATASET=data/full
# DATASET=data/crop

for featurefile in ${DATASET}/tsne/*.h5; do
    python scripts/tsne_map.py ${featurefile} --perplexity 40 --bordersize 8
done

