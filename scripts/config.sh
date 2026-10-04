#!/bin/bash
# Site-specific settings. Every launcher sources this file, and all scripts are
# meant to be run from the repository root. Override any value by exporting it
# before launching, or edit the defaults here.

# Repository root and storage for datasets / checkpoints.
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_DATASET="${SSD_DATASET:-$SSD_MOUNT/dataset}"
export SSD_WEIGHTS="${SSD_WEIGHTS:-$SSD_MOUNT/weights}"

# Python environment (see scripts/miscellaneous/install.sh).
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-flame-moe}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Weights & Biases. Leave WANDB_ENTITY empty to log under your default entity.
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-moe}"

# Only needed by the singularity-based dataset scripts (scripts/dataset/*_local.slurm).
export SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-}"
export SINGULARITY_OVERLAY="${SINGULARITY_OVERLAY:-}"
