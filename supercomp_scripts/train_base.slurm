#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=01:00:00
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1

# Run the container
apptainer shell  --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/model/lightning_logs:/app/lightning_logs --nv ${VSC_SCRATCH}/model/env.sif<<EOF

# Change to /app directory
cd /app

# Run the Python script
python python train.py --config "base_model" -c --data-file data/mtl_5fold_cv/all_data.csv --train-i data/mtl_5fold_cv/train_0.csv --val-i data/mtl_5fold_cv/val_0.csv --test-i data/mtl_5fold_cv/test_0.csv --hpt-config hpt/mtl_hpt_supervised_none.csv --hpt-id 21 --vocab-file data/mtl_5fold_cv/all_data_vocab.p

EOF