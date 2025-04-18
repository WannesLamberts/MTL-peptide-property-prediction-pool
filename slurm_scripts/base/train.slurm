#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=02:00:00
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1

NAME=$1
TRAIN=$2


# Run the container
apptainer shell --bind ${VSC_SCRATCH}/model/MTL-peptide-property-prediction-base:/app --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/model/lightning_logs:/app/lightning_logs --nv ${VSC_SCRATCH}/model/environment/environment.sif<<EOF

# Change to /app directory
cd /app

# Run the Python script
python train.py --config $NAME -p own --checkpoint-id 0 -c --data-file "data/train.parquet" --train-i "data/$TRAIN" --val-file "data/val.parquet" --test-file "data/test.parquet" --hpt-config "hpt/mtl_hpt_finetune_own.csv" --hpt-id 27 --vocab-file "data/vocab.p" --lookup "data/lookup.parquet"

EOF