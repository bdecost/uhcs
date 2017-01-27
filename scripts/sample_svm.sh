#!/bin/bash

# run final SVM 10x 10-fold cross-validation

# DATASET=data/full
DATASET=data/crop

echo "bow features"
for featurefile in ${DATASET}/features/*bow*.h5; do
    sbatch scripts/sample_svm.py ${featurefile} --kernel chi2 -C 1 --n-repeats 5 -d 64;
done

echo "vlad features"
for featurefile in ${DATASET}/features/*vlad*.h5; do
    sbatch scripts/sample_svm.py ${featurefile} --kernel linear -C 1 --n-repeats 5 -d 64;
done

sbatch scripts/sample_svm.py ${DATASET}/features/raw.h5 --kernel linear -C 1  --n-repeats 5 -d 64;
