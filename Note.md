## TODO

- [x] set logging in bash, one have to build the directory fist.

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

* **SAVE LIMIT_RAM VISION AS PKL FILE.**

* LIMIT_RAM example: 1.pkl

```python
{
'gt_classes': array([1382, 1383, ..., 4090, 4091], dtype=int32), 
'flipped': False, 
'gt_phrases': {1536: [3, 10, 20, 8, 6, 2, 9], 3584: [36, 38, 29, 17, 2, 37], ...},
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
'gt_phrases': {1536: [3, 10, 20, 8, 6, 2, 9], 3584: [36, 38, 29, 17, 2, 37], ...},
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
'gt_overlaps': <262x2 sparse matrix of type '<type 'numpy.float32'>'
               with 262 stored elements in Compressed Sparse Row format>}
}
```

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
 'gt_phrases': {1536: [3, 10, 20, 8, 6, 2, 9], 3584: [36, 38, 29, 17, 2, 37], ...}
}             
```
  
