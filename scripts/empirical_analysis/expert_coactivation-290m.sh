#!/bin/bash
# Compute the expert coactivation for FLAME-MoE-290M-1.3B

#SBATCH --job-name=expert-coactivation-290m
#SBATCH --output=logs/%x/%j.log
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Set up CUDA environment
# Load site configuration (run from the repository root)
source scripts/config.sh

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate conda environment
source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

# use local actives (no GCP download)
# MODEL_NAME: the folder name under actives/ (e.g., "290m", "290m-dro-activation-...")
# ACTIVES_DIR: override to use different base directory (default: $SSD_MOUNT/actives)
# NUM_EXPERTS: number of experts in the model (default: 64, use 32 for 32-expert models)
# CHECKPOINT: specific checkpoint to process (e.g., "16000"), or "all" for all checkpoints
export MODEL_NAME=${MODEL_NAME:-290m}
export ACTIVES_DIR=${ACTIVES_DIR:-$SSD_MOUNT/actives}
export NUM_EXPERTS=${NUM_EXPERTS:-64}
export CHECKPOINT=${CHECKPOINT:-all}
bash scripts/empirical_analysis/modules/expert_coactivation_step1.sh

echo "Processing expert coactivation for: $MODEL_NAME"
echo "Actives directory: $ACTIVES_DIR/$MODEL_NAME"
echo "Checkpoint filter: $CHECKPOINT"

# process each layer inside each checkpoint (model-specific actives path)
if [ "$CHECKPOINT" = "all" ]; then
    SEARCH_PATH="$ACTIVES_DIR/$MODEL_NAME"
    FIND_DEPTH="-mindepth 2 -maxdepth 2"
else
    SEARCH_PATH="$ACTIVES_DIR/$MODEL_NAME/$CHECKPOINT"
    FIND_DEPTH="-mindepth 1 -maxdepth 1"
fi

find $SEARCH_PATH $FIND_DEPTH -type d | while read -r actives_path; do
    results_path=results/expert-coactivation/$MODEL_NAME/$(basename $(dirname $actives_path))/$(basename $actives_path).pkl
    echo "Processing $actives_path -> $results_path"
    python3 scripts/empirical_analysis/modules/expert_coactivation_step2.py --actives-path $actives_path --results-path $results_path
done

echo "Co-activation analysis complete. Results saved to: results/expert-coactivation/$MODEL_NAME/"
