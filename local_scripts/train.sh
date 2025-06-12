#!/bin/bash


NAME=$1
TRAIN=$2

python train.py --config $NAME -p own --checkpoint-id 0 -c --data-file "data/train.parquet" --train-i "data/$TRAIN" --val-file "data/val.parquet" --test-file "data/test.parquet" --hpt-config "hpt/mtl_hpt_finetune_own.csv" --hpt-id 27 --vocab-file "data/vocab.p" --lookup "data/lookup.parquet"
read -p "Press Enter to exit..."
