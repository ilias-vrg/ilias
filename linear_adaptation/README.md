# Linear Adaptation

This code implements the training of linear adaptation layers as explained in the ILIAS paper.
Those layers are linear transformations used to project off-the-shelf embeddings on a new space for better retrieval on the ILIAS dataset.
This layer is trained with embeddings from samples of the multi-domain UnED dataset, which is specific to fine-grained and instance-level retrieval.
The code also supports the extraction the UnED features for the model to be adapted.

## Setup

### Prerequisites

Ensure the following are installed on your system:

* Python (>= 3.10)
* PyTorch

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
pip install -r linear_adaptation/requirements.txt
```

### Download UnED

* Download the [UnED dataset](https://cmp.felk.cvut.cz/univ_emb/#dataset) by following the instructions [here](https://github.com/nikosips/Universal-Image-Embeddings?tab=readme-ov-file#dataset-preparation), following the first 3 steps of `dataset preparation`. UnED root path provided in the feature extraction step in this script should point to the `images` directory of the UnED dataset.

## Usage

* Run the `feature_extraction/extract.py` script to extract embeddings for the (our subset of) UnED training set, for a specific pretrained model.
These embeddings will be used to train the linear adaptation layer.

```bash
python feature_extraction/extract.py \
   --partition uned \
   --dataset_dir </path/to/uned/images/> \
   --hdf5_dir </path/to/hdf5/uned/embeddings/> \
   --framework <codebase> \
   --model <model_name> \
   --resolution <resolution> \
   --selected_ids misc/uned_paths_1M.txt
```

* Run the `train_lin_adapt.py` script to train the linear adaptation layer which is (optionally) used in the kNN search, with the desired arguments:

```bash
python linear_adaptation/train_lin_adapt.py \
   --uned_features_path </path/to/hdf5/uned/embeddings/> \
   --uned_info_path misc/uned_paths_1M.txt \
   --adaptation_layer_dim <output_dims> \
   --num_epochs <number_of_epochs> \
   --batch_size <batch_size> \
   --lr <learning_rate> \
   --save_dir </save_path/to/lin_adapt_layer>
```

### Example

* To reproduce the training of the linear adaptation layer with the extracted UnED features for SigLIP, run the following:

```bash
python feature_extraction/extract.py \
    --partition uned \
    --dataset_dir /path/to/uned/ \
    --hdf5_dir /path/to/hdf5 \
    --framework timm \
    --model vit_large_patch16_siglip_384.webli \
    --resolution 512 \
    --selected_ids misc/uned_paths_1M.txt
```

```bash
python linear_adaptation/train_lin_adapt.py \
    --uned_features_path /path/to/hdf5/features_uned.hdf5 \
    --uned_info_path misc/uned_paths_1M.txt \
    --adaptation_layer_dim 512 \
    --num_epochs 2 \
    --batch_size 128 \
    --lr 0.001 \
    --save_dir /path/to/lin_adapt_layers/siglip.pth
```

### Output

The output of running the above two scripts is a .pth file containing the (pytorch) checkpoint of the linear adaptation layer, ready to be ported to the knn search that uses is to transform the features.

### Notes

* You can find the trained layers for all models used in out paper [here](https://vrg.fel.cvut.cz/ilias_data/lin_adapt_layers/).

* The embeddings for the samples of the subset of the UnED dataset that are used to train the linear adaptation layer (for the most popular models used in our benchmark) can be found [here](https://vrg.fel.cvut.cz/ilias_data/features/).
