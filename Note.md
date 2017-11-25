## TODO

- [x] set logging in bash, one have to build the directory fist.
- [ ] check out the dropout using in fc7
- [ ] consider add dropout in embedding layer

## Data Preprocessing

* The total number of raw image data: 108249
* The total number of image descriptions: 108078. With a json file corresponding to a image which has the same name
(i.e. the id number)  
* Dataset typo: `bckground`

```python
{"region_id": 1936, "width": 40, "height": 38, "image_id": 1, 
"phrase": "bicycles are seen in the bckground", "y": 320, "x": 318, 
"phrase_tokens": ["bicycles", "are", "seen", "in", "the", "bckground"]}
```
* Read a gt_region json take `1 ms`.
* `x_gt_region/id.json` has the format:

```python
{
"regions":[ ... 
            {"region_id": 1382, "width": 82, "height": 139, "image_id": 1, 
            "phrase": "the clock is green in colour", "y": 57, "x": 421, 
            "phrase_tokens": ["the", "clock", "is", "green", "in", "colour"]},
            ... 
           ],
"path": "/home/joe/git/VG_raw_data/img_test/1.jpg", 
"width": 800, 
"id": 1, 
"height": 600
}
           
```

### roi_db

* **Add "gt_phrases" to every roidb, both with LIMIT_RAM version or UNLIMIT_RAM version.
* **SAVE LIMIT_RAM VISION AS PKL FILE.**

* LIMIT_RAM example: 1.pkl

```python
{
'gt_classes': array([1382, 1383, ..., 4090, 4091], dtype=int32), 
'flipped': False, 
'gt_phrases': [[4, 33, 6, 25, 20, 144], [167, 6, 30, 4, 11], [7, 6, 21, 72],...],
'boxes': array([[421,  57, 503, 196],
                [194, 372, 376, 481],
                [241, 491, 302, 521],
                ...], dtype=uint16),
'seg_areas': array([  11620.,   20130.,    1922., ...], dtype=float32), 
'gt_overlaps': <262x2 sparse matrix of type '<type 'numpy.float32'>'
               with 262 stored elements in Compressed Sparse Row format>}
}
.update
{
'width': 800, 
'max_classes': array([1, 1, 1, ...]),
'image': u'/home/joe/git/VG_raw_data/img_test/1.jpg', 
'max_overlaps': array([ 1.,  1.,  1.,  1.,  1., ...]),
'height': 600, 
'image_id': 1
}
```

**NOTE: `gt_phrases` add 1 before saving it. ** 
```python
# increment the stream -- 0 will be the EOS character
stream = [s + 1 for s in stream]
```

* LIMIT_RAM example: 1_flip.pkl

```python
{
'gt_classes': array([1382, 1383, ..., 4090, 4091], dtype=int32), 
'flipped': True, 
'gt_phrases': [[4, 33, 6, 25, 20, 144], [167, 6, 30, 4, 11], [7, 6, 21, 72],...],
'boxes': array([[296,  57, 378, 196],
                [423, 372, 605, 481],
                [497, 491, 558, 521],
                ...], dtype=uint16),
'seg_areas': array([  11620.,   20130.,    1922., ...], dtype=float32), 
'gt_overlaps': <262x2 sparse matrix of type '<type 'numpy.float32'>'
               with 262 stored elements in Compressed Sparse Row format>}
}
.update
{
'width': 800, 
'max_classes': array([1, 1, 1, ...]),
'image': u'/home/joe/git/VG_raw_data/img_test/1.jpg', 
'max_overlaps': array([ 1.,  1.,  1.,  1.,  1., ...]),
'height': 600, 
'image_id': '1_flip'
}
```

* UNLIMIT_RAM example: pre_gt_roidb.pkl

```python
{
'gt_classes': array([1382, 1383, ..., 4090, 4091], dtype=int32), 
'flipped': False, 
'boxes': array([[421,  57, 503, 196],
                [194, 372, 376, 481],
                [241, 491, 302, 521],
                ...], dtype=uint16),
'seg_areas': array([  11620.,   20130.,    1922., ...], dtype=float32), 
'gt_phrases': [[4, 33, 6, 25, 20, 144], [167, 6, 30, 4, 11], [7, 6, 21, 72],...],
'gt_overlaps': <262x2 sparse matrix of type '<type 'numpy.float32'>'
               with 262 stored elements in Compressed Sparse Row format>}
}
```

**DO NOT HAVE ALL_PHRASES**
* UNLIMIT_RAM exampl: 

```python
{1536: [3, 10, 20, 8, 6, 2, 9], 3584: [36, 38, 29, 17, 2, 37], ...}
```

## LIMIT RAM

* Return `roidb` is a path to the saved pkls.
* TRAIN.USE_FLIPPED = True
**NEEDS TO:**  
1. add USE_FLIPPED to be True? added on 10.18.17
2. `rdl_roidb.prepare_roidb` method to process data. 
3. filter out the invalid roi before

* add self.image_index to visual_genome class for filterd indexes. **Update:** change to self._image_index.

* finish roidatalayer, **Read image in BGR order**. Example of `data.forward()` with 1.jpg
  * gt_phrases shape: num_regions x 10(max_words)

```json
{
 'gt_boxes': array([[  2.66399994e+02,   5.12999992e+01,   3.40200012e+02,
                       1.76399994e+02,   1.38200000e+03],
                    [  3.80700012e+02,   3.34799988e+02,   5.44500000e+02,
                       4.32899994e+02,   1.38300000e+03], ...], dtype=float32),
 'data': array([[[[-14.9183712 , -28.93106651,  -9.59885979],
                  [-18.94306374, -32.79834747, -13.62355137],
                  [-11.24244404, -24.31069756,  -5.66058874],...]]], dtype=float32),
 'im_info': array([[ 540.        ,  720.        ,    0.89999998]], dtype=float32),
 'gt_phrases': array([[  4,  33,   6, ...,   0,   0,   0],
                      [167,   6,  30, ...,   0,   0,   0],
                      [  7,   6,  21, ...,   0,   0,   0],...]
}             
```

## Sentence data layer

Output of first 3 regions of 1.jpg:

```python
# length of labels, i.e. number of regions: 262
# sentence data layer input (first 3)
1382.0 [  4  33   6  25  20 144   0   0   0   0]
1383.0 [167   6  30   4  11   0   0   0   0   0]
1384.0 [ 7  6 21 72  0  0  0  0  0  0]
# sentence data layer output (first 3)
# input sentence
[[   1.    4.   33.    6.   25.   20.  144.    0.    0.    0.    0.]
 [   1.  167.    6.   30.    4.   11.    0.    0.    0.    0.    0.]
 [   1.    7.    6.   21.   72.    0.    0.    0.    0.    0.    0.]]
target sentence
[[   1.    4.   33.    6.   25.   20.  144.    2.    0.    0.    0.    0.]
 [   1.  167.    6.   30.    4.   11.    2.    0.    0.    0.    0.    0.]
 [   1.    7.    6.   21.   72.    2.    0.    0.    0.    0.    0.    0.]]
# cont sentence
[[ 0.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.]
 [ 0.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.]
 [ 0.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.]]
# cont bbox
[[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]]
```

## Embedding

* index `0` will be the `<pad>` character
* index `1` will be the `<SOS>` character
* index `2` will be the `<EOS>` character


## Architecture test

image:  (1, 540, 720, 3)  
head:   (1, 34, 45, 1024)  
rpn:    (1, 34 ,45, 512)  
rpn_cls_score:  (1, 34, 45, 24) #2 x 3 x 4  
rpn_cls_score_reshape:  (1, 34x12, 45, 2)  
rpn_cls_prob_reshape:  (1, 34x12, 45, 2)   
rpn_cls_prob:  (1, 34, 45, 24) #2 x 3 x 4  
rpn_bbox_pred:  (1, 34, 45, 48)  
anchors  ==> (18360, 4)  
**proposal layer**  
proposal_rois:  (9, 5) #due to NMS, it's a heavy reduce to proposals.  
proposal_rpn_scores: (9, 1)  
**anchor target layer**  
rpn_labels:  (1, 1, 408, 45)  
rpn_bbox_targets: (1, 34, 45, 48)  
rpn_bbox_inside_weights: (1, 34, 45, 48)  
rpn_bbox_outside_weights: (1, 34, 45, 48)  
**proposal_targets_single_class_layer**  
make sure a fixed number of regions are sampled.  
rois: (256, 5)  
labels   ==> (256,)  
clss  ==> (256,)  
phrases  ==> (256, 10)  
bbox_targets  ==> (256, 4)  
bbox_inside_weights  ==> (256, 4)  
bbox_outside_weights ==> (256, 4)   
**RPN**  
pool5  ==> (256, 7, 7, 1024)  
fc7    ==> (256, 2048)  
name: fc7_before_pool               ==> (256, 7, 7, 2048)  
cls_prob   ==> (256, 2)  
**sentence data layer**  
input_sentence  ==> (256, 11)  
target_sentence  ==> (256, 12)  
cont_bbox   ==> (256, 12)  
cont_sentence  ==> (256, 12)  
**embed_caption_layer**  
name: embedding               ==> (10003, 512)  
name: embed_input_sentence               ==> (256, 11, 512)  
name: fc8               ==> (256, 512)  
name: im_context               ==> (256, 1, 512)  
name: im_concat_words               ==> (256, 12, 512)  
name: captoin_outputs               ==> (256, 12, 512)  
name: loc_outputs               ==> (256, 12, 512)  
name: bbox_pred               ==> (256, 4)  
name: predict_caption               ==> (256, 12, 10003)  


