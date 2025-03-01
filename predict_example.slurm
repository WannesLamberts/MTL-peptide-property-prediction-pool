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
python predict.py

EOF

