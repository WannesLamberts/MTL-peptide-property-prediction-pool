#!/bin/bash


NAME=$1
DIR=$2

# Run the Python script
python train.py --config $NAME -p own --checkpoint-id 0 -c --train-file "data/$DIR/all.parquet" --val-file "data/val.parquet" --test-file "data/test.parquet" --hpt-config "hpt/mtl_hpt_finetune_own.csv" --hpt-id 27 --vocab-file "data/vocab.p" --lookup "data/$DIR/lookup.parquet"
read -p "Press Enter to exit..."
