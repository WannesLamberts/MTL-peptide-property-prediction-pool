#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=01:00:00
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1

# Run the container
apptainer shell --bind ${VSC_SCRATCH}/model/MTL-peptide-property-prediction-pool:/app --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/model/lightning_logs:/app/lightning_logs --nv ${VSC_SCRATCH}/model/environment/environment.sif<<EOF

# Change to /app directory
cd /app

# Run the Python script
python train.py --config "pool_1k_rsplit" -p own --checkpoint-id 0 -c --data-file data/1k_rsplit/all_data.csv --train-i data/1k_rsplit/train_0.csv --val-i data/1k_rsplit/val_0.csv --test-i data/1k_rsplit/test_0.csv --hpt-config hpt/mtl_hpt_supervised_none.csv --hpt-id 21 --vocab-file data/1k_rsplit/vocab.p --lookup data/1k_rsplit/loopkup.parquet

EOF