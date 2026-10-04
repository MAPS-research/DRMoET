#!/bin/bash
# Sweep aggressive DRO hyperparameters with activation weighting
#
# Grid:
#   beta: 0.9, 0.95
#   eta: 0.01, 0.05, 0.1
#   norm: l1, l2
# Total: 12 runs

# Fixed config (same as flame-moe-290m-dro-activation.sh)

# Load site configuration (run from the repository root)
source scripts/config.sh

export NUM_LAYERS=9
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=5472
export MOE_FFN_HIDDEN_SIZE=704
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=8
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=2
export TRAIN_ITERS=5473
export SAVE_INTERVAL=540
export EVAL_INTERVAL=540
export USE_DRO=true
export DRO_ALPHA=1.0
export LOG_DRO_SIGNALS=true
export DRO_ROUTER_GRAD_CLIP=0.0
export DRO_MU_UPDATE_CLIP=0.0
export USE_QK_NORM=false
export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"

# Calculate EVAL_ITERS (constant for all runs)
WORLD_SIZE=16  # 4 nodes * 4 GPUs
DATA_PARALLEL_SIZE=$((WORLD_SIZE / (1 * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE)))
CURRENT_EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))
export EVAL_ITERS=$((100 * CURRENT_EFFECTIVE_BATCH_SIZE / 32))

echo "=== DRO Activation Sweep ==="
echo "eta: 0.01, 0.05, 0.1"
echo "beta: 0.9, 0.95"
echo "norm: l1, l2"
echo "Total: 12 runs"
echo ""

# Sweep parameters
for BETA in 0.9 0.95; do
  for ETA in 0.01 0.05 0.1; do
    for NORM in l1 l2; do
      export DRO_BETA=$BETA
      export DRO_ETA=$ETA
      export DRO_ACTIVATION_NORM_TYPE=$NORM
      export MODEL_SIZE_ID="290m-dro-activation-beta${BETA}-eta${ETA}-norm${NORM}"
      export SSD_WEIGHTS="$SSD_MOUNT/weights/$MODEL_SIZE_ID"

      echo "Submitting: $MODEL_SIZE_ID"
      sbatch --job-name=$MODEL_SIZE_ID --nodes=4 scripts/training/flame-moe_local_dro_activation.slurm
      sleep 2  # Avoid overwhelming scheduler
    done
  done
done

echo ""
echo "All 12 jobs submitted. Check with: squeue -u \$USER"
