# Re-ranking with local features

This code implements re-ranking of the initial global ranking with AMES model and local features.

## Setup

### Prerequisites

Ensure the following are installed on your system:

* Python (>= 3.10)
* PyTorch

### Installation

* Clone this repo

   ```bash
   git clone git@github.com:ilias-vrg/ilias.git
   cd ilias/local_reranking
   ```

* [Optional] Create a new environment for the project:

   ```bash
   conda create -n ilias python=3.10
   conda activate ilias
   ```

* Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

* [Optional] Generate a shortlist as `image_ids` in the form of txt file with newline-separated image paths. These shortlist files can be created via a helper script that accepts the global ranking similarity file created by the `knn_search` module:

```bash
python scripts/get_shortlist.py \
  --similarity_file </path/to/results/file/> \
  --save_dir </path/to/shortlist>
```

* Run the `feature_extraction/extract.py` script to extract local features via `detector` pooling that uses a pre-trained local feature detector.
* The code currently only supports extraction of DINOv2-base local features.
* The features are automatically binarized to limit the required memory.

```bash
python feature_extraction/extract.py \
  --partition <ilias_partition> \
  --dataset_dir </path/to/images/or/tars/> \
  --hdf5_dir </path/to/hdf5> \
  --framework fb \
  --model dinov2_vitb14_reg \
  --pooling detector \
  --resolution 728 \
  --store_fp16 false \
  --comp_fp16 true \
  --hdf5_name local_features \
  --image_ids </path/to/shortlist>
```

* After pre-extracting the features, run the `rerank.py` script with the following arguments:

```bash
python local_reranking/rerank.py
   --similarity_file </path/to/results/file> \
   --query_hdf5 /path/to/hdf5/features_image_queries.hdf5 \
   --positive_hdf5 /path/to/hdf5/features_positive.hdf5 \
   --distractor_hdf5 </path/to/distractor/hdf5> \
   --total_distractors <number_of_hdf5_files> \
   --batch_size <num_pairs_processed_at_once> \
   --save_dir </path/to/similarities> \
   --save_name <name_of_similarity_file> \
   --save_as {json, pickle} \
   --use_gpu
```

* The script supports two formats for storing the similarity scores, same as the `knn_search` module.

### Example

* The run arguments can look for example as:

```bash
python local_reranking/rerank.py \
   --similarity_file similarities.json \
   --query_hdf5 /path/to/hdf5/local_features_image_queries.hdf5 \
   --positive_hdf5 /path/to/hdf5/local_features_positives.hdf5 \
   --distractor_hdf5 /path/to/hdf5/local_features_distractors_{idx}.hdf5 \
   --total_distractors 10 \
   --k 1000 \
   --batch_size 256 \
   --num_workers 4 \
   --lamb 0.35 \
   --temp 0.1 \
   --save_path /path/to/similarities/ \
   --save_name local_similarities \
   --save_as json \
   --use_gpu
```

* In the `distractor_hdf5` argument, the `{idx}` indicates the location in the path name that corresponds to the index of the distractor hdf5 files. It is replaced during loading with the corresponding index values according to `total_distractors`.

### Output

* The script supports two formats for storing the refined similarity scores, similar to [knn_search](knn_search#output).

### Notes

* We provide the extracted local binary features for shortlists based on two global models with linear adaptation (SigLIP-L@384 and PE-L@336) [here](https://vrg.fel.cvut.cz/ilias_data/features/).

* The images of these shortlists (the union of the top-1000 YFCC100M images over all queries) are also available as tars [here](https://vrg.fel.cvut.cz/ilias_data/shortlists/), see [download](../download/README.md#shortlists), so local features can be extracted for them without downloading the 100M distractors.

* We also provide the estimated similarities refined with AMES for the two shortlists [here](https://vrg.fel.cvut.cz/ilias_data/similarities/).

___

> [!WARNING]
> Please note that the AMES model used for re-ranking in the ILIAS paper differs from the version in the official [AMES repository](https://github.com/pavelsuma/ames). To ensure exact reproduction of our results and for archival purposes, we host the [specific model](https://vrg.fel.cvut.cz/ilias_data/ames/) version separately within the ILIAS resources.
