#!/bin/bash

DATASET=$1
PREDICT=$2

# Run the Python script
python get_encodings.py --run "lightning_logs/CONFIG=mtl_5foldcv_pretrain_0,TASKS=CCS_iRT,MODE=pretrain,PRETRAIN=none,LR=0.0001940554482365,BS=1024,OPTIM=adam,LOSS=mae,CLIP=False,ACTIVATION=gelu,SCHED=warmup,SIZE=180,NUMLAYERS=9/version_0" --all_data_file "data/$DATASET/all_data.csv" --predict_i "data/$DATASET/$PREDICT" --out_file "data/$DATASET/lookup.parquet"
read -p "Press Enter to exit..."
