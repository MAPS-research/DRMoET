#!/bin/bash

#SBATCH --job-name=flame-moe-local
#SBATCH --output=logs/%x/%j.log
#SBATCH --time=0-14:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --signal=B:TERM@300
#SBATCH --open-mode=append

# Set up environment variables
# Load site configuration (run from the repository root)
source scripts/config.sh

export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_DATASET="$SSD_MOUNT/dataset/tokenized/EleutherAI/pythia-12b"
# SSD_WEIGHTS is now set by the release script to include model size
# export SSD_WEIGHTS="$SSD_MOUNT/weights"  # This is now set in the release script

# Training configuration
export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="${RDZV_ENDPOINT:-$(hostname):8000}"

# Weights & Biases Configuration
# WANDB_ENTITY: Your wandb username or team name
export WANDB_ENTITY="${WANDB_ENTITY:-}"
# WANDB_PROJECT: The project name within your entity
export WANDB_PROJECT="${WANDB_PROJECT:-pretrain_drmoet}"
# WANDB_RUN_GROUP: Groups related runs together (e.g., same experiment)
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$SLURM_JOB_NAME}"
# WANDB_NAME: Individual run name (defaults to SLURM job ID)
export WANDB_NAME="${WANDB_NAME:-$SLURM_JOB_ID}"

# Use local tokenized dataset
export TRAIN_DATASET="$SSD_DATASET/tokenized/EleutherAI/pythia-12b"
export TRAIN_WEIGHTS="$SSD_WEIGHTS/$SLURM_JOB_NAME/$SLURM_JOB_ID"

# Create necessary directories
mkdir -p $SSD_WEIGHTS

# Run training directly (no download step needed since data is already local)
srun -W 0 scripts/training/modules/flame-moe_step2_local_aux_loss_free.sh