#!/bin/bash
# Compute expert utilization for FLAME-MoE-290M models
# Reuses the same actives data as coactivation analysis

#SBATCH --job-name=expert-utilization-290m
#SBATCH --output=logs/%x/%j.log
#SBATCH --time=0-4:00:00
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

# Configuration
# MODEL_NAME: folder name under actives/ (e.g., "290m-seed3407-32e")
# ACTIVES_DIR: base directory for actives (default: ${SSD_MOUNT}/actives)
# NUM_EXPERTS: number of experts (default: 32 for these models)
# CHECKPOINT: specific checkpoint or "all"
export MODEL_NAME=${MODEL_NAME:-290m-seed3407-32e}
export ACTIVES_DIR=${ACTIVES_DIR:-${SSD_MOUNT}/actives}
export NUM_EXPERTS=${NUM_EXPERTS:-32}
export CHECKPOINT=${CHECKPOINT:-16000}

echo "============================================"
echo "Expert Utilization Analysis"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Actives: $ACTIVES_DIR/$MODEL_NAME"
echo "Num experts: $NUM_EXPERTS"
echo "Checkpoint: $CHECKPOINT"
echo "============================================"

mkdir -p logs/expert-utilization-290m

# Process each layer for the specified checkpoint(s)
if [ "$CHECKPOINT" = "all" ]; then
    CHECKPOINTS=$(ls $ACTIVES_DIR/$MODEL_NAME/)
else
    CHECKPOINTS=$CHECKPOINT
fi

for ckpt in $CHECKPOINTS; do
    echo ""
    echo ">>> Processing checkpoint: $ckpt"

    # Process each layer
    for layer_dir in $(ls -d $ACTIVES_DIR/$MODEL_NAME/$ckpt/*/); do
        layer=$(basename $layer_dir)
        results_path="results/expert-utilization/$MODEL_NAME/$ckpt/layer_$layer.json"

        echo "  Layer $layer -> $results_path"
        python3 scripts/empirical_analysis/modules/expert_utilization_step2.py \
            --actives-path "$layer_dir" \
            --results-path "$results_path"
    done

    # Aggregate results for THIS checkpoint
    echo "  >>> Aggregating checkpoint $ckpt..."
    python3 << EOF
import json
import numpy as np
from pathlib import Path

model_name = "$MODEL_NAME"
checkpoint = "$ckpt"
num_experts = $NUM_EXPERTS

results_dir = Path(f"results/expert-utilization/{model_name}/{checkpoint}")
if not results_dir.exists():
    print(f"No results found in {results_dir}")
    exit(1)

# Aggregate across layers
total_counts = np.zeros(num_experts, dtype=np.float64)

for json_file in results_dir.glob("layer_*.json"):
    with open(json_file) as f:
        data = json.load(f)
    total_counts += np.array(data["counts"])

# Compute aggregate stats
total = total_counts.sum()
dist = total_counts / (total + 1e-10)
cv = np.std(dist) / np.mean(dist) * 100
sorted_c = np.sort(total_counts)
n = len(total_counts)
gini = (2 * np.sum((np.arange(1, n+1) * sorted_c))) / (n * total) - (n + 1) / n
entropy = -np.sum(dist * np.log(dist + 1e-10))

aggregate = {
    "model": model_name,
    "checkpoint": checkpoint,
    "num_experts": num_experts,
    "total_tokens": float(total),
    "counts": total_counts.tolist(),
    "cv_percent": float(cv),
    "gini": float(gini),
    "entropy": float(entropy),
    "max_entropy": float(np.log(num_experts)),
    "entropy_ratio": float(entropy / np.log(num_experts)),
    "max_min_ratio": float(total_counts.max() / (total_counts.min() + 1e-10)),
}

output_path = results_dir / "aggregate.json"
with open(output_path, 'w') as f:
    json.dump(aggregate, f, indent=2)

print(f"  CV: {cv:.2f}%, Gini: {gini:.4f}, Saved: {output_path}")
EOF
done

echo ""
echo "============================================"
echo "Analysis complete!"
echo "Results: results/expert-utilization/$MODEL_NAME/"
echo "============================================"
