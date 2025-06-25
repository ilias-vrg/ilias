#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=30
#SBATCH --output=./logs/%j.out

# Set the environment variables
ILIAS_ROOT=$1
YFCC_ROOT=$2
FEATURE_ROOT=$3
FRAMEWORK=$4
MODEL=$5
POOLING=$6
RESOLUTION=$7
IS_TEXT=$8

srun -l python feature_extraction/extract.py \
    --partition image_queries \
    --dataset_dir $ILIAS_ROOT \
    --hdf5_dir $FEATURE_ROOT/$MODEL/ \
    --framework $FRAMEWORK \
    --model $MODEL \
    --resolution $RESOLUTION \
    --pooling $POOLING

srun -l python feature_extraction/extract.py \
    --partition positives \
    --dataset_dir $ILIAS_ROOT \
    --hdf5_dir $FEATURE_ROOT/$MODEL/ \
    --framework $FRAMEWORK \
    --model $MODEL \
    --resolution $RESOLUTION \
    --pooling $POOLING

srun -l python feature_extraction/extract.py \
    --partition distractors \
    --dataset_dir $YFCC_ROOT \
    --hdf5_dir $FEATURE_ROOT/$MODEL/ \
    --framework $FRAMEWORK \
    --model $MODEL \
    --resolution $RESOLUTION \
    --pooling $POOLING \
    --start_tar 0 \
    --total_tars 10 \

if [ "$IS_TEXT" = "true" ]; then
    srun -l python feature_extraction/extract.py \
        --partition text_queries \
        --dataset_dir $ILIAS_ROOT \
        --hdf5_dir $FEATURE_ROOT/$MODEL/ \
        --framework text \
        --model $MODEL
fi