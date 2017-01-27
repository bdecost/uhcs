#!/bin/bash

DATASET=data/full

# PCA, MDS, LLE, Isomap, SpectralEmbedding
method=LLE

echo "bow features"
for featurefile in ${DATASET}/features/*bow*.h5; do
    sbatch scripts/manifold_embed.py ${featurefile} --kernel chi2 --method ${method};
done

echo
echo "vlad features"

for featurefile in ${DATASET}/features/*vlad*.h5; do
    sbatch scripts/manifold_embed.py ${featurefile} --kernel linear --method ${method};
done

sbatch scripts/manifold_embed.py ${DATASET}/features/raw.h5 --kernel linear --method ${method};
