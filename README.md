# Densecap-tensorflow

Implementation of CVPR2017 paper: [Dense captioning with joint inference and visual context](https://arxiv.org/abs/1611.06949) by **Linjie Yang, Kevin Tang, Jianchao Yang, Li-Jia Li**

**WITH CHANGES:**  
1. Borrow the idea of [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/abs/1611.01462), and tied word vectors and word classfiers during captioning.
2. Initialize Word Vectors and Word Classifers with pre-trained [glove](https://nlp.stanford.edu/projects/glove/) word vectors with dimensions of 300.
3. Change the backbone of the framework to ResNet-50.
4. Add "Limit_RAM" mode when praparing training date since my computer only has RAM with 8G.

**Special thanks to [valohai](https://valohai.com/) for offering computing resource.**

## NOTE
* After 1 epoch(80000 iters) of training with randomly initialized word vectors(512d), it achieves **mAP 6.509**. Training curve and samples are coming soon.
* After 1 epoch(75000) of training with pre-trianed glove word vectors(300d), it got **mAP 5.5** nearly.
* The complete training process will take almost **10 days** with the computation I have access to, and I just trained 1 epoch to varify the framework for now.
* The scripts should be compatible with both python 2.X and 3.X. Although I built it under python 2.7.
* Tested on Ubuntu 16.04, tensorflow 1.4, CUDA 8.0 and cudnn 6.0, with GPU Nvidia gtx 1060(LOL...).

## Dependencies

To install required python modules by:

```commandline
pip install -r lib/requirements.txt
```

**For evaluation, one also need:**  
* java 1.8.0
* python 2.7(according to [coco-caption](https://github.com/tylin/coco-caption)

To install java runtime by:  
```commandline
sudo apt-get install openjdk-8-jre
```

## Preparing data

Firstly, check `lib/config.py` for `LIMIT_RAM` option. If one has RAM `less than 16G`, I recommend setting `__C.LIMIT_RAM = True`(default True).
* If `LIMIT_RAM = True`, setting up the data path in `info/read_regions.py` accordingly, and run the script with python. Then it will dump `regions` in `REGION_JSON` directory. It will take time to process more than 100k images, so be patient.
* In `lib/preprocess.py`, set up data path accordingly. After running the file, it will dump `gt_regions` of every image respectively to `OUTPUT_DIR` as `directory` or just a big `json` file.

## Compile local libs

```shell
$ cd root/lib
$ make
```

## Train

1. Add or modify configurations in `root/scripts/dense_cap_config.yml`
2. 


## TODO:

- [x] preprocessing dataset.
- [x] roi_data_layer & get data well prepared for feeding.
- [x] proposal layer
- [x] sentense data layer
- [x] embedding layer
- [x] get loc loss and caption loss
- [x] overfit a mini-batch
- [x] context fusion

## References

* The Faster-RCNN framework inherited from repo [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) by [endernewton](https://github.com/endernewton)
* The official repo of [densecap](https://github.com/linjieyangsc/densecap)
* [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/abs/1611.01462)
