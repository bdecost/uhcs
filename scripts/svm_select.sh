#!/bin/bash

# DATASET=data/full
DATASET=data/crop

echo "bow features"
for featurefile in ${DATASET}/features/*bow*.h5; do
    sbatch scripts/svm_param_select.py ${featurefile} --kernel chi2 --n-per-class 50 --n-repeats 10 --seed 0;
done

echo
echo "vlad features"

for featurefile in ${DATASET}/features/*vlad*.h5; do
    sbatch scripts/svm_param_select.py ${featurefile} --kernel linear --n-per-class 50 --n-repeats 10 --seed 0;
done
