#!/bin/bash
# Compute the router saturation for FLAME-MoE-290M-DRO-Activation

#SBATCH --job-name=router-saturation-290m-dro-activation
#SBATCH --output=logs/%x/%j.log

#SBATCH --time=14-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1536G
#SBATCH --cpus-per-task=208
#SBATCH --gres=gpu:8

# Model identifier for DRO activation method
# Load site configuration (run from the repository root)
source scripts/config.sh

export MODEL_NAME="290m-dro-activation-54730-step-beta0.999-eta0.001-alpha1.0-norml2_fine_grained"

# No step1 needed - actives are stored locally from capture-290m.sh

# Process each top-k for FLAME-MoE-290M-DRO-Activation
for moe_router_topk in 6 4 2 1; do
    for layer_number in {2..9}; do
        actives_pattern="$SSD_MOUNT/actives/$MODEL_NAME/*/$layer_number"
        results_path=results/router-saturation/$MODEL_NAME/$layer_number/$moe_router_topk.pkl
        python3 scripts/empirical_analysis/modules/router_saturation_step2.py \
            --moe-router-topk $moe_router_topk \
            --actives-pattern "$actives_pattern" \
            --results-path $results_path
    done
done
