#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=02:00:00
#SBATCH --partition=ampere_gpu
#SBATCH --gpus-per-node=1

NAME=$1
TRAIN=$2
HPT_ID=$3

# Run the container
apptainer shell --bind ${VSC_SCRATCH}/model/MTL-peptide-property-prediction-pool:/app --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/model/lightning_logs:/app/lightning_logs --nv ${VSC_SCRATCH}/model/environment/environment.sif<<EOF

# Change to /app directory
cd /app

# Run the Python script
python train.py --config $NAME -p own --checkpoint-id 0 -c --data-file "data/dataset.parquet" --train-i "data/$TRAIN" --val-i "data/val.csv" --test-i "data/test.csv" --hpt-config "hpt/hpt_class.csv" --hpt-id $HPT_ID --vocab-file "data/vocab.p" --bs 2048 --type base
EOF