#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=02:00:00
#SBATCH --partition=ampere_gpu
#SBATCH --gpus-per-node=1


# Run the container
apptainer shell --bind ${VSC_SCRATCH}/model/MTL-peptide-property-prediction-pool:/app --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/model/lightning_logs:/app/lightning_$
# Change to /app directory
cd /app

# Run the Python script
python param_tune.py --checkpoint-id 0 --data-file "data/test/dataset.parquet" --train-i "data/test/train.csv" --val-i "data/test/val.csv" --vocab-file "data/vocab.p" --lookup "data/test/$EOF