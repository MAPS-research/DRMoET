#!/bin/bash
# Capture the router traces for FLAME-MoE-290M-1.3B.

#SBATCH --job-name=capture-290m
#SBATCH --output=logs/%x/%j.log
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Change to project directory
# Set up CUDA environment
# Load site configuration (run from the repository root)
source scripts/config.sh

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate conda environment
source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

# model architecture for FLAME-MoE-290M
# NUM_EXPERTS: 64 (default) or 32 for 32-expert models
export NUM_LAYERS=${NUM_LAYERS:-9}
export HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-5472}
export MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-704}
export MOE_LAYER_FREQ=${MOE_LAYER_FREQ:-"[0]*1+[1]*8"}
export NUM_EXPERTS=${NUM_EXPERTS:-64}
export MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-6}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-2}  # Must divide evenly into GPU count
export TRAIN_ITERS=${TRAIN_ITERS:-5473}
export RDZV_BACKEND="c10d"

# where to load the pretrained weights and dataset (local paths)
# MODEL_NAME: the folder name under weights/ (e.g., "290m", "290m-dro-activation-...")
# WEIGHTS_DIR: override to use different base directory (default: $SSD_WEIGHTS)
# CHECKPOINTS: space-separated list of steps, or "all" for all checkpoints
# OUTPUT_DIR: where to save actives (default: $SSD_MOUNT, point at a large filesystem to avoid file limits)
export MODEL_NAME=${MODEL_NAME:-290m}
export WEIGHTS_DIR=${WEIGHTS_DIR:-$SSD_WEIGHTS}
export OUTPUT_DIR=${OUTPUT_DIR:-$SSD_MOUNT}
export TRAIN_WEIGHTS=$WEIGHTS_DIR/$MODEL_NAME
export TRAIN_DATASET=$SSD_DATASET/tokenized/EleutherAI/pythia-12b
export CHECKPOINTS=${CHECKPOINTS:-all}

srun scripts/empirical_analysis/modules/capture_step1.sh

# trace the router for each iteration
PORT=29500
for item in $(ls -d $TRAIN_WEIGHTS/iter_* | sort -r); do
    name=$(basename $item)
    step=$((10#${name#iter_}))

    # Skip if CHECKPOINTS is not "all" and step not in list
    if [ "$CHECKPOINTS" != "all" ]; then
        if ! echo "$CHECKPOINTS" | grep -qw "$step"; then
            echo "Skipping checkpoint $step (not in CHECKPOINTS)"
            continue
        fi
    fi

    echo "Capturing $step on port $PORT ..."
    export RDZV_ENDPOINT="localhost:$PORT"
    export EACT_SAVE=$OUTPUT_DIR/actives/$MODEL_NAME/$step
    export TIDS_SAVE=$OUTPUT_DIR/samples
    echo $step > $TRAIN_WEIGHTS/latest_checkpointed_iteration.txt
    srun scripts/empirical_analysis/modules/capture_step2.sh
    PORT=$((PORT + 1))
    sleep 5  # Give time for port to be released
done

srun scripts/empirical_analysis/modules/capture_step3.sh
