#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=02:00:00
#SBATCH --partition=ampere_gpu
#SBATCH --gpus-per-node=1


# Run the container
apptainer shell --bind ${VSC_SCRATCH}/model/MTL-peptide-property-prediction-pool:/app --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/model/lightning_logs:/app/lightning_logs --nv ${VSC_SCRATCH}/model/environment/environment.sif<<EOF

# Change to /app directory
cd /app

# Run the Python script
python param_tune.py --checkpoint-id 0 --data-file "data/dataset.parquet" --train-i "data/train.csv" --val-i "data/val.csv" --vocab-file "data/vocab.p" --lookup "data/lookup.parquet" --bs 2048 --time 3 --type pool --epochs 50 --amount 6
EOF