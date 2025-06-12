# Download the ILIAS dataset

The ILIAS dataset is hosted on our servers and is accessible via [this portal](https://vrg.fel.cvut.cz/ilias_data/). It consists of two main parts [ILIAS core](https://vrg.fel.cvut.cz/ilias_data/ilias_core.tar) (the collected images) and [YFCC100M](https://vrg.fel.cvut.cz/ilias_data/yfcc100m/). The YFCC100M images are chunked into shards of 100K images each. The total amount needed to store the full dataset (ILIAS core + YFCC100M) is 6.3TB.

## Downloading
* Run this script to download the full dataset.

```bash
bash download_ilias.sh
```

* You can provide arguments to the script to specify the directory to store ILIAS (default is `./ilias`), the number of YFCC100M shards to download (default is `0`, i.e. all shards), and the number of retries per shard if the downloaded shard is corrupted (default is `1`).

```bash
bash download_ilias.sh <ilias_dir> <num_of_yfcc100m_shards> <num_of_retries>
```

e.g. 

```bash
bash download_ilias.sh /path/to/ilias 2 3
```

## Dataset structure

### folder structure
* The ILIAS object and their corresponding images are stored in the following file structure.
 
```
ilias_core
└── dummy_name_xxx
    ├── query
    │   ├── Qxxx_yy.jpg
    │   ├── Qxxx_yy_bbox.txt
    │   ├── Qxxx_yy_url.txt (optional)
    │   └── Txxx.txt
    └── pos
        ├── Pxxx_zz.jpg
        ├── Pxxx_zz_bbox.png
        └── Pxxx_zz_url.txt (optional)
```

where `xxx` is the id of the object instance. `Q` indicates image queries with `yy` as their corresponding id. `T` indicates text queries. `P` indicates positive images with `zz` as their corresponding id. All images are stored in `jpg` format and accompanied by a txt file with `bbox.txt` extension containing the bounding box coordinates of the query object. Images downloaded from the internet are accompanied by a file with `url.txt` extension that encloses the original image URL.

* For example, the first and last objects in the dataset

```
ilias_core
├── bold_bimp_000
│   ├── query
│   │   ├── Q000_00.jpg
│   │   ├── Q000_00_bbox.txt
│   │   └── T000.txt
│   └── pos
│       ├── P000_00.jpg
│       ├── P000_00_bbox.txt
│       ├── ...
│       ├── P000_08.jpg
│       └── P000_08_bbox.txt
├── ...
└── zippy_zootlenibble_999
    ├── query
    │   ├── Q999_00_bbox.txt
    │   ├── Q999_00.jpg
    │   └── T999.txt
    └── pos
        ├── P999_00.jpg
        ├── P999_00_bbox.txt
        ├── ...
        ├── P999_04.jpg
        └── P999_04_bbox.txt
```

### bounding boxes
* The bounding box files contain several lines that indicate the location of the query object in the images. More than one bounding box may be provided for an image. The locations of bounding boxes are infered by the pixel coordinates of the `top-left corner` and the `height` and `width` of the box. The bounding boxes are stored in the following format.

```
<x of top-left point> <y of top-left point> <width> <height>
```

### taxonomy

* The compiled taxonomy for the ILIAS object instances is shared in the [`taxonomy.json`](../misc/taxonomy.json) file. It is organized into a three-level hierarchy: coarse-level categories (e.g. `product`, `landmark`, `art`), mid-level categories (e.g. `food`, `architecture`, `sculpture`), and fine-level categories (e.g., `cake`, `building`, `3d print`). Each fine-level category contains the object instance identifiers. The `other` category includes objects that do not fit into any category. 

## License

All images captured by us or external collaborators are shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en). The full list of captured images can be found [here](../misc/ilias_core_captured.txt). All external collaborators have signed consent agreements to share their images under the aforementioned license. The images downloaded from the internet are shared under their original license, which can be found in their original URL.