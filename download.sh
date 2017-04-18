#!/bin/bash
# usage: `bash download.sh`

# get data from NIST for this project
# http://hdl.handle.net/11256/940
NIST_DATASET=11256/940
NIST_DATASET_URL=https://materialsdata.nist.gov/dspace/xmlui/bitstream/handle/${NIST_DATASET}

DATADIR=uhcsdata

echo "download toplevel files"
for archivefile in README.md setup.sh; do
    curl ${NIST_DATASET_URL}/${archivefile} -o ${archivefile}
done

for archivefile in uhcsdb.zip tools.zip; do
    curl ${NIST_DATASET_URL}/${archivefile} -o ${archivefile}
    unzip ${archivefile}
done

echo "download data files into DATADIR=${DATADIR}"
mkdir -p ${DATADIR}
touch ${DATADIR}/__init__.py

for archivefile in microstructures.sqlite models.py; do
    curl ${NIST_DATASET_URL}/${archivefile} -o ${DATADIR}/${archivefile}
done

for archivefile in micrographs.zip representations.zip embed.zip figures.zip; do
    curl ${NIST_DATASET_URL}/${archivefile} -o ${DATADIR}/${archivefile}
    unzip ${DATADIR}/${archivefile} -d ${DATADIR}
done
