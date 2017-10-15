## Data Preprocessing

* The total number of raw image data: 108249
* The total number of image descriptions: 108078. With a json file corresponding to a image which has the same name
(i.e. the id number)  
* Dataset typo: `bckground`

```json
{"region_id": 1936, "width": 40, "height": 38, "image_id": 1, 
"phrase": "bicycles are seen in the bckground", "y": 320, "x": 318, 
"phrase_tokens": ["bicycles", "are", "seen", "in", "the", "bckground"]}
```
* Read a gt_region json take `1 ms`.
* `x_gt_region/id.json` has the format:

```json
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
