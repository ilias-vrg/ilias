# kNN Search

This code implements k-Nearest Neighbors (kNN) search and evaluation on ILIAS dataset. kNN search pipeline compares query embeddings against a database of embeddings to perfrom retrieval. 

## Setup

### Prerequisites

Ensure the following are installed on your system:

* Python (>= 3.10)
* PyTorch
* FAISS

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
pip install -r knn_search/requirements.txt
```

## Usage

* Run the `search.py` script with the desired arguments:

```bash
python knn_search/search.py \
   --query_hdf5 </path/to/query/hdf5> \
   --positive_hdf5 </path/to/positive/hdf5> \
   --distractor_hdf5 </path/to/distractor/hdf5> \
   --k <number_of_kNN> \
   --total_distractors <number_of_hdf5_files> \
   --save_dir </path/to/similarities> \
   --save_name <name_of_similarity_file> \
   --save_as {json, pickle} \
   --lin_adopt_path </path/to/lin_adopt_layer>
```

### Example

* To runs kNN search with extracted features, run the following:

```bash
python knn_search/search.py \
   --query_hdf5 /path/to/hdf5/features_image_queries.hdf5 \
   --positive_hdf5 /path/to/hdf5/features_positive.hdf5 \
   --distractor_hdf5 /path/to/hdf5/features_distractors_{idx}.hdf5 \
   --k 1000 \
   --total_distractors 100 \
   --save_path /path/to/similarities/ \
   --save_name similarities \
   --save_as json \
   --lin_adopt_path /path/to/lin_adapt_layers/vit_large_patch16_siglip_384.webli.pth
```

* In the `distractor_hdf5` argument, the `{idx}` indicates the location in the path name that corresponds to the index of the distractor hdf5 files. It is replaced during loading with the corresponding index values according to `total_distractors`.

* Search only in the mini-ILIAS by providing to the `selected_ids` argument the path to [this file](https://vrg.fel.cvut.cz/ilias_data/image_ids/mini_ilias_yfcc100m_ids.txt). It is not necessary to provide this argument if the extracted features contain only mini-ILIAS distractors.

* You can also provide a linear adaptation layer using the `lin_adopt_path`. You can find the trained layers for all models used in out paper [here](https://vrg.fel.cvut.cz/ilias_data/lin_adapt_layers/).

### Output

* The script supports two formats for storing the similarity scores:

Json
```
{
   query_1: {
      retrieved_db_1: 0.9,
      retrieved_db_2: 0.8,
      retrieved_db_2: 0.7,
      ... },
   query_2: {
      retrieved_db_1: 0.7,
      retrieved_db_2: 0.6,
      retrieved_db_2: 0.5,
      ... },
   ...
}
```

or 

Pickle
```
{
   "query_ids": <list_of_queries>,
   "db_ids": <list_of_dbs>,
   "sims": <matrix_of_similarities>,
   "ranks": <matrix_of_indices>,
}
```

### Estimated similarities

* We provide the estimated similarities based some selected models [here](https://vrg.fel.cvut.cz/ilias_data/similarities/).

* [Contact us](mailto:kordogeo@gel.cvut.cz?subject=[ILIAS]%20request%20for%20similarities) to provide similarities based on models not included in the above link.