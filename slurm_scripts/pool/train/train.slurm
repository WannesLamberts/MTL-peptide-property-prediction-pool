#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=01:00:00
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1

FILENAMES=$1
SPLIT=$2

# Run the container
apptainer shell --bind ${VSC_SCRATCH}/model/MTL-peptide-property-prediction-pool:/app --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/model/lightning_logs:/app/lightning_logs --nv ${VSC_SCRATCH}/model/environment/environment.sif<<EOF

# Change to /app directory
cd /app

# Run the Python script
python train.py --config "pool_$1_$2" -p own --checkpoint-id 0 -c --data-file "data/200_filenames/all_data.csv" --train-i data/200_filenames/filenames=$1,amount=None,split=$2/train.csv --val-i data/200_filenames/filenames=$1,amount=None,split=$2/val.csv --test-i data/200_filenames/filenames=$1,amount=None,split=$2/test.csv --hpt-config hpt/mtl_hpt_finetune_own.csv --hpt-id 27 --vocab-file data/200_filenames/vocab.p --lookup data/200_filenames/filenames=$1,amount=None,split=$2/lookup.parquet

EOF