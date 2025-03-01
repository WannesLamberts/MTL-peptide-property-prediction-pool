#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=01:00:00
#SBATCH --partition=zen2

# Run the container
apptainer shell --bind ${VSC_SCRATCH}/model/data:/app/data --bind ${VSC_SCRATCH}/lightning_logs:/app/model/lightning_logs --nv ${VSC_SCRATCH}/model/my_container.sif <<EOF
# Check if /app directory exists
if [ ! -d "/app" ]; then
    echo "/app directory not found in container"
    exit 1
fi

# Change to /app directory
cd /app

# Check if predict.py exists
if [ ! -f "predict.py" ]; then
    echo "predict.py not found!"
    exit 1
fi

# Run the Python script
python train.py --config "mtl_5foldcv_finetune_own_0" -p own --checkpoint-id 0 -c --data-file data/sample_1k/all_data.csv --train-i data/sample_1k/train_0.csv --val-i data/sample_1k/val_0.csv --test-i data/sample_1k/test_0.csv --hpt-config hpt/mtl_hpt_finetune_own.csv --hpt-id 27 --vocab-file data/ceder/all_data_vocab.p

EOF