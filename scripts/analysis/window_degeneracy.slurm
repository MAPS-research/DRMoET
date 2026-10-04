#!/bin/bash
#SBATCH --job-name=win-degen
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# Load site configuration (run from the repository root)
source scripts/config.sh

source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p analysis/degeneracy
python scripts/analysis/window_degeneracy.py \
  "$(cat scripts/analysis/all_s2orc_domains.txt)" \
  analysis/degeneracy/s2orc.json
