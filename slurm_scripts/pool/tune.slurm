#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=02:00:00
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1

DATASET=$1
CONFIG=$2
ID=$3

# Run the container
apptainer shell --bind ${VSC_SCRATCH}/model/MTL-peptide-property-prediction-pool:/app --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/model/lightning_logs:/app/lightning_logs --nv ${VSC_SCRATCH}/model/environment/environment.sif<<EOF

# Change to /app directory
cd /app

# Run the Python script
python param_tune.py--config "scheduler" -i $ID -p own --checkpoint-id 0 --data-file "data/$DATASET/all_data.csv" --train-i "data/$DATASET/train.csv" --val-i "data/$DATASET/val.csv" --vocab-file "data/$DATASET/vocab.p" --lookup "data/$DATASET/lookup.parquet"
EOF