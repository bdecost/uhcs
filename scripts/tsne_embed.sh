#!/bin/bash

# DATASET=data/full
# DATASET=data/cropped
DATASET=data/crop

echo "bow features"
for featurefile in ${DATASET}/features/*bow*.h5; do
    sbatch scripts/tsne_embed.py ${featurefile} --kernel chi2  --n-repeats 10;
done

echo
echo "vlad features"

for featurefile in ${DATASET}/features/*vlad*.h5; do
    sbatch scripts/tsne_embed.py ${featurefile} --kernel linear --n-repeats 10;
done

sbatch scripts/tsne_embed.py ${DATASET}/features/raw.h5 --kernel linear --n-repeats 10;
