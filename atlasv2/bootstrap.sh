#!/bin/bash
# =============================
# How to use slurm to run GPU
# =============================
# How to run: 
# > sbatch bootstrap.sh {your_python_script} {your_args}
#
# > squeue (to watch your jobs)
# > skill [id] (to cancel your job)
#
# To run a local bash
# > srun $(command -v apptainer || command -v singularity) exec --nv "$SIF" bash

# -J: Name of the job
#SBATCH -J bootstrap_cuda

# -p Cluster (atlasv2_mia_gpu01_1t4|atlasv2_mia_gpu02_4t4|atlasv2_mia_cpu01)
#SBATCH -p atlasv2_mia_gpu01_1t4

# --gres gpu How many GPUs are we requiring
#SBATCH --gres=gpu:1

# --time How much time our job will be running
#SBATCH --time=00:40:00

# --ntasks Number of tasks
#SBATCH --ntasks=1

# --cpus-per-task How many CPUs for tasks.
#SBATCH --cpus-per-task=2

# --mem How many RAM do we need
#SBATCH --mem=12G

# -- Redirect output and error to files
# You can set to out_%j.txt if you want to add the ID of the job
#SBATCH -o out.txt
#SBATCH -e err.txt

# Configure
# AppTainer Container
SIF="/software/singularity/Informatica/mia-dlpln2-apptainer/mia_dlpln_parte2_1.0.sif"


# Bind args to pass to the python script
if [[ $# -lt 1 ]]; then
echo "Uso: sbatch $0 <script.py> [args...]" >&2
exit 1
fi
PYFILE="$1"
shift
PYARGS=("$@")


# To display pipeerrors and to log more easy
# @link https://stackoverflow.com/questions/68465355/what-is-the-meaning-of-set-o-pipefail-in-bash-script
set -euo pipefail


# Create the user directory in scratch
SCRATCH_DIR="/scratch/$USER"
mkdir -p "$SCRATCH_DIR"
chmod 700 "$SCRATCH_DIR"


# Grant permissions to cache folders
mkdir -p "$SCRATCH_DIR/.hf/hub" "$SCRATCH_DIR/.hf/transformers" \
         "$SCRATCH_DIR/.cache" "$SCRATCH_DIR/.torch" "$SCRATCH_DIR/tmp"
chmod 700 "$SCRATCH_DIR" "$SCRATCH_DIR/.hf" "$SCRATCH_DIR/.hf/hub" "$SCRATCH_DIR/.hf/transformers" \
          "$SCRATCH_DIR/.cache" "$SCRATCH_DIR/.torch" "$SCRATCH_DIR/tmp"


# Move everything to scratch
export SCRATCH_DIR="/scratch/$USER"
export HF_HOME="$SCRATCH_DIR/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$SCRATCH_DIR/.hf/datasets"
export HF_HUB_CACHE="$SCRATCH_DIR/.hf/hub"
export TORCH_HOME="$SCRATCH_DIR/.torch"
export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
export TMPDIR="$SCRATCH_DIR/tmp"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"


# BLAS threads limited to the assigned CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}


# HF Token


# To get info about the GPU
nvidia-smi || true


# Run
srun $(command -v apptainer || command -v singularity) exec --nv \
  --bind /scratch:/scratch \
  "$SIF" \
  python -u "$PYFILE" "${PYARGS[@]}"
  
