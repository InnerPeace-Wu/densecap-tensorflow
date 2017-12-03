#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED='True'

DATASET='visual_genome_1.2'
NET='res50'
ckpt_path='/home/joe/git/slim_models'
data_dir='/home/joe/git/visual_genome'

case $DATASET in
   visual_genome)
    TRAIN_IMDB="vg_1.0_train"
    TEST_IMDB="vg_1.0_val"
    PT_DIR="dense_cap"
    ITERS=$2
#    FINETUNE_AFTER1=200000
#    FINETUNE_AFTER2=100000
#    ITERS1=400000
#    ITERS2=300000
    ;;
  visual_genome_1.2)
    TRAIN_IMDB="vg_1.2_train"
    TEST_IMDB="vg_1.2_val"
    PT_DIR="dense_cap"
    ITERS=$2
#    FINETUNE_AFTER1=200000
#    FINETUNE_AFTER2=100000
#    ITERS1=400000
#    ITERS2=300000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

if [ -d '/valohai/outputs' ]; then
    LOG="/valohai/outputs/${NET}_${TRAIN_IMDB}_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
    # prapare data
    apt-get -y update
    pip install -r requirements.txt
    cd /valohai/inputs
    tar -xvzf ./vg_data/visual_genome.tar.gz
    mkdir ./images
    unzip -xvzf image_1/images.zip -d ./images
    unzip -xvzf image_2/images2.zip -d ./images
    ckpt_path='/valohai/inputs/resnet'
    data_dir='/valohai/inputs/visual_genome'
    cd /valohai/repository
    cd lib
    make
    cd ..
else
    LOG="tests/logs/${NET}_${TRAIN_IMDB}_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
fi

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py \
    --weights ${ckpt_path}/${NET}.ckpt \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${ITERS} \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET}

if [ -d '/valohai/outputs']; then
    tar -czvf /valohai/outputs/output.tar.gz ./output
fi

