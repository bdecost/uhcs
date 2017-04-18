#!/bin/bash
# usage: `bash download.sh`

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
