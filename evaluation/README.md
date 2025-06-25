# ILIAS evaluation

This code implements the evaluation on ILIAS dataset via mean Average Precision at k (mAP@k).

## Setup

### Prerequisites

Ensure the following are installed on your system

* Python (>= 3.10)

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
pip install -r evaluation/requirements.txt
```

* Make sure that the `image_ids` folder exists in the dataset directory. Otherwise, run the following command:

```bash
bash download_ids.sh /path/to/dataset/
```

## Usage

* Run the `evaluation.py` script providing the file of similarities in either `json` or `pickle` format, and the dataset directory:

```bash
python evaluation/evaluate.py \
  --dataset_dir /path/to/dataset/ \
  --similarity_file <similarity_file> \
  --k <value_of_k>
```

* The output of the script is the mAP@k achieved with the provided `similarity_file`.

* Providing a file path name to the `result_file` argument will generate a pickle file containing detailed results. Specifically, it contains average precision per query, oracle score, performance per (sub)category (cf. Figure 5, F in the paper), and performance per scale/clutter groups (cf. Fig. 7).