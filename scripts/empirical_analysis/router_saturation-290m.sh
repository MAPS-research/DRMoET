#!/bin/bash
# Compute the router saturation for FLAME-MoE-290M-1.3B

#SBATCH --job-name=router-saturation-290m
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Activate conda environment
# Load site configuration (run from the repository root)
source scripts/config.sh

source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

# Configuration variables:
# MODEL_NAME: the folder name under actives/ (e.g., "290m", "290m-dro-activation-...")
# ACTIVES_DIR: override to use different base directory (default: $SSD_MOUNT/actives)
# MOE_ROUTER_TOPK: space-separated list of top-k values to analyze (default: "6 4 2 1")
export MODEL_NAME=${MODEL_NAME:-290m}
export ACTIVES_DIR=${ACTIVES_DIR:-$SSD_MOUNT/actives}
export MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-"6 4 2 1"}

echo "Processing router saturation for: $MODEL_NAME"
echo "Actives directory: $ACTIVES_DIR/$MODEL_NAME"

# Process each top-k value
for moe_router_topk in $MOE_ROUTER_TOPK; do
    for layer_number in {2..9}; do
        actives_pattern="$ACTIVES_DIR/$MODEL_NAME/*/$layer_number"
        results_path=results/router-saturation/$MODEL_NAME/$layer_number/$moe_router_topk.pkl
        echo "Processing layer $layer_number, top-k $moe_router_topk -> $results_path"
        python3 scripts/empirical_analysis/modules/router_saturation_step2.py \
            --moe-router-topk $moe_router_topk \
            --actives-pattern "$actives_pattern" \
            --results-path $results_path
    done
done

echo "Router saturation analysis complete. Results saved to: results/router-saturation/$MODEL_NAME/"
