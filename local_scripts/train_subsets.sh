#!/bin/bash

DATASET=$1
FILENAMES=$2

# Run the Python script
python train.py --config "pool_${DATASET}_${FILENAMES}" -p own --checkpoint-id 0 -c --data-file "data/$DATASET/all_data.csv" --train-i "data/$DATASET/train_subsets/train_${FILENAMES}.csv" --val-i "data/$DATASET/val.csv" --hpt-config "hpt/mtl_hpt_finetune_own.csv" --hpt-id 27 --vocab-file "data/$DATASET/vocab.p" --lookup "data/$DATASET/lookup.parquet"
read -p "Press Enter to exit..."
