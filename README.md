# densecap-tensorflow

## NOTE
* The scripts should be compatible with both python 2.X and 3.X. Although I built it under python 2.7.

## TODO:

- [x] preprocessing dataset

## Dependencies

```commandline
pip install -r lib/requirements.txt
```

## Preparing data

* Firstly, check `lib/config.py` for `LIMIT_RAM` option. If one has RAM `less than 16G`, I recommend 
setting `__C.LIMIT_RAM = True`(default True).
    * If `LIMIT_RAM = True`, setting up the data path in `info/read_regions.py` accordingly, and run 
    the script with python. Then it will dump 
    `regions` in `REGION_JSON` directory. It will take time to process more than 100k images, so be patient.
    * In `lib/preprocess.py`, set up data path accordingly. After running the file, it will dump `gt_regions` of
    every image respectively to `OUTPUT_DIR` as `directory` or just a big `json` file.
