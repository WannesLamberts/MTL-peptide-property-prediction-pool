#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=01:00:00
#SBATCH --partition=zen2

apptainer shell ${VSC_SCRATCH}/MTL-peptide-property-prediction/my-apptainer-image.sif
