# Feature Extraction

This code implements feature extraction for the different partitions of ILIAS dataset. It supports several codebases and feature extraction based on raw images or tars of images. Additionally, it supports feature extraction for global and local descriptors for images and text descriptions.

## Setup

### Prerequisites

Ensure the following are installed on your system

* Python (>= 3.10)
* PyTorch
* Torchvision

### Installation

* Clone this repo

```bash
git clone git@github.com:ilias-vrg/ilias.git
cd ilias
```

* [Optional] Create a new environment for the project

```bash
python -m venv ilias
source ilias/bin/activate
```

or

```bash
conda create -n ilias python=3.10
conda activate ilias
```

* Install the required packages:

```bash
pip install -r feature_extraction/requirements.txt
```

## Usage

### Image feature extraction

* Run the `extract.py` script with the desired arguments:

```bash
python feature_extraction/extract.py \
  --partition <ilias_partition> \
  --dataset_dir </path/to/images/or/tars/> \
  --hdf5_dir </path/to/hdf5> \
  --framework <codebase> \
  --model <model_name> \
  --resolution <resolution>
```

* You need to specify the `framework` and `model` that you want to use to extract features. Also, you can specify the `resolution` of the image that is used to resize the image based on the selected `transforms`. In the paper, we use the `RESIZE_LARGE` transform, which resizes images based on their larger side. This is the default option.

### Text feature extraction

* Run the `extract.py` script with the desired arguments:

```bash
python feature_extraction/extract.py \
  --partition text_queries \
  --dataset_dir </path/to/ilias/core/> \
  --hdf5_dir </path/to/hdf5> \
  --framework text \
  --model <model_name>
```

* You need to specify the `model` that you want to use for feature extraction.

### Example with ILIAS-core

* To extract features for all partitions of ILIAS dataset using SigLIP, run the following:

Feature extraction for queries

```bash
python feature_extraction/extract.py \
  --partition image_queries \
  --dataset_dir /path/to/images \
  --hdf5_dir /path/to/hdf5 \
  --framework timm \
  --model vit_large_patch16_siglip_384.webli \
  --resolution 512
```

Feature extraction for positives

```bash
python feature_extraction/extract.py \
  --partition positives \
  --dataset_dir /path/to/images \
  --hdf5_dir /path/to/hdf5 \
  --framework timm \
  --model vit_large_patch16_siglip_384.webli \
  --resolution 512
```

Feature extraction for text descriptions

```bash
python feature_extraction/extract.py \
  --partition text_queries \
  --dataset_dir /path/to/images \
  --hdf5_dir /path/to/hdf5 \
  --framework text \
  --model vit_large_patch16_siglip_384.webli
```

* These commands will generate three HDF5 files in the `/path/to/hdf5` directory with names `features_image_queries.hdf5`, `features_positives.hdf5`, and `features_text_queries.hdf5` containing the features of the queries, positive images, and instance descriptions in ILIAS-core, respectively.

### Example with YFCC100M

* For the distrators, the extraction can be done in batches by providing the `start_tar`, the index of the tar to start the extraction, and the `total_tars`, the total number of tar to be extracted. Each tar in the [`yfcc100m`](https://vrg.fel.cvut.cz/ilias_data/yfcc100m/) folder contains 100k images.

#### **full ILIAS**

* For feature extraction of the first 1M distractors (10 yfcc100m tars), run the command

```bash
python feature_extraction/extract.py \
  --partition distractors \
  --dataset_dir /path/to/tars \
  --hdf5_dir /path/to/hdf5 \
  --start_tar 0 \
  --total_tars 10 \
  --framework timm \
  --model vit_large_patch16_siglip_384.webli \
  --resolution 512
```

* Increase `start_tar` to extract the next 1M distractors, and run the command

```bash
python feature_extraction/extract.py \
  --partition distractors \
  --dataset_dir /path/to/tars \
  --hdf5_dir /path/to/hdf5 \
  --start_tar 10 \
  --total_tars 10 \
  --framework timm \
  --model vit_large_patch16_siglip_384.webli \
  --resolution 512
```

* Continue until you process all tar in the `yfcc100m` folder.

* Running these commands will produce a several HDF5 files in the `/path/to/hdf5` directory. Each file is named `features_distractors_<index>.hdf5` (for example, `features_distractors_00000.hdf5`, `features_distractors_00001.hdf5`, etc.), and contains the feature vectors for the corresponding set of distractors.

#### **mini-ILIAS**

* To extract features only for mini-ILIAS, provide to the `selected_ids` argument the path to [this file](https://vrg.fel.cvut.cz/ilias_data/image_ids/mini_ilias_yfcc100m_ids.txt).

```bash
python feature_extraction/extract.py \
  --partition distractors \
  --dataset_dir /path/to/tars \
  --hdf5_dir /path/to/hdf5 \
  --framework timm \
  --model vit_large_patch16_siglip_384.webli \
  --resolution 512 \
  --selected_ids /path/to/mini-ilias-ids
```

### Model settings

* The parameters used for all models can be found in [`models.sh`](scripts/models.sh).

* An example slurm script to extract features for ILIAS-core and 1M of YFCC100m images is [`run_slurm.sh`](scripts/run_slurm.sh).

* Edit the above scripts providing the local paths that the dataset is stored.

### Extracted feaures

* We provide the extracted features for some selected models [here](https://vrg.fel.cvut.cz/ilias_data/features/).

* [Contact us](mailto:kordogeo@gel.cvut.cz?subject=[ILIAS]%20request%20for%20features) to provide features for models not included in the above link.