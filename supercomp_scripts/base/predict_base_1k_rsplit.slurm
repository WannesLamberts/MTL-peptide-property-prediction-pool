#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=01:00:00
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1

# Run the container
apptainer shell --bind ${VSC_SCRATCH}/model/MTL-peptide-property-prediction-base:/app --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/model/lightning_logs:/app/lightning_logs --nv ${VSC_SCRATCH}/model/environment/environment.sif<<EOF

# Change to /app directory
cd /app

# Run the Python script
python predict.py --run "lightning_logs/CONFIG=base_1k_rsplit,TASKS=CCS_iRT,MODE=supervised,PRETRAIN=own,LR=0.0003105497384738,BS=1024,OPTIM=adamw,LOSS=mae,CLIP=False,ACTIVATION=gelu,SCHED=warmup_decay_cos,SIZE=768,NUMLAYERS=12/version_0" --all_data_file "data/1k_rsplit/all_data.csv" --predict_i "data/1k_rsplit/test_0.csv"

EOF