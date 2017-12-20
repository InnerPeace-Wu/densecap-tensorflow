#!/usr/bin/env bash

# Preprocessing data in valohai computing platform.
# This script may out of date. #2017.12.20
set -e
set -x

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    #statements
    key="$1"

    case $key in
        -vs|--version)
        VERSION=$2
        shift
        shift
        ;;
        -p|--path)
        IN_PATH=$2
        shift
        shift
        ;;
        -od|--output_dir)
        OUTPUT_DIR=$2
        shift
        shift
        ;;
        -mw|--max_words)
        MAX_WORDS=$2
        shift
        shift
        ;;
        *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done


if [ -d "/valohai/inputs" ]; then
    # apt-get -y update
    # apt-get -y install python-pip
    pip install -r requirements.txt
    cd /valohai/inputs
    mkdir ${VERSION}
    unzip image_meta/image_data.json.zip -d ./${VERSION}
    unzip regions/region_descriptions.json.zip -d ./${VERSION}
    cd /valohai/repository/lib
    time python2 preprocess.py --version ${VERSION} \
        --path ${IN_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --max_words ${MAX_WORDS}

    tar -czvf /valohai/outputs/visual_genome.tar.gz ${OUTPUT_DIR}
    # comment it if one already have data stored in S3
    mv regions/region_descriptions.json.zip /valohai/outputs
fi



