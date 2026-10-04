#!/bin/bash
# Capture expert activations on domain-specific data for specialization analysis
#
# Usage:
#   DOMAIN=code MODEL_NAME=290m-seed3407-32e CHECKPOINT=16000 sbatch capture_domain.sh
#
# This will capture activations and save to:
#   ${SSD_MOUNT}/actives/domain/{model_name}/{checkpoint}/{domain}/

#SBATCH --job-name=capture-domain
#SBATCH --output=logs/%x/%j.log
#SBATCH --time=0-2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load site configuration (run from the repository root)
source scripts/config.sh

set -euo pipefail

# Set up CUDA environment
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate conda environment
source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

# Configuration
export DOMAIN=${DOMAIN:-code}
export MODEL_NAME=${MODEL_NAME:-290m-seed3407-32e}
export CHECKPOINT=${CHECKPOINT:-16000}
export DOMAIN_DATA_DIR=${DOMAIN_DATA_DIR:-data/domain_eval}
export ACTIVES_BASE=${ACTIVES_BASE:-${SSD_MOUNT}/actives/domain}
export NUM_EXPERTS=${NUM_EXPERTS:-32}
export SEQ_LEN=${SEQ_LEN:-2048}
export BATCH_SIZE=${BATCH_SIZE:-4}

# Paths
DOMAIN_DATA="$DOMAIN_DATA_DIR/$DOMAIN.jsonl"
ACTIVES_DIR="$ACTIVES_BASE/$MODEL_NAME/$CHECKPOINT/$DOMAIN"

echo "============================================"
echo "Domain Capture for Specialization Analysis"
echo "============================================"
echo "Domain: $DOMAIN"
echo "Model: $MODEL_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Domain data: $DOMAIN_DATA"
echo "Output: $ACTIVES_DIR"
echo "============================================"

# Check domain data exists
if [ ! -f "$DOMAIN_DATA" ]; then
    echo "ERROR: Domain data not found: $DOMAIN_DATA"
    echo "Run: python scripts/empirical_analysis/prepare_domain_data.py --domains $DOMAIN"
    exit 1
fi

mkdir -p "$ACTIVES_DIR"
mkdir -p logs/capture-domain

# Find checkpoint path
if [[ "$MODEL_NAME" == *"dro"* ]]; then
    WEIGHTS_DIR="${SSD_MOUNT}/weights/290m-dro-activation-beta0.999-eta0.001-alpha1.0-norml2_fine_grained_32e"
else
    WEIGHTS_DIR="${SSD_MOUNT}/weights/290m-seed3407-32e"
fi

CHECKPOINT_PATH="$WEIGHTS_DIR/iter_$(printf '%07d' $CHECKPOINT)"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "Checkpoint path: $CHECKPOINT_PATH"

# Run capture
# This uses the same capture mechanism as capture-290m.sh but with domain-specific data
python3 scripts/empirical_analysis/modules/capture_domain_step2.py \
    --domain-data "$DOMAIN_DATA" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --actives-dir "$ACTIVES_DIR" \
    --num-experts "$NUM_EXPERTS" \
    --seq-len "$SEQ_LEN" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "============================================"
echo "Capture complete!"
echo "Actives saved to: $ACTIVES_DIR"
echo "============================================"
