#!/bin/bash
# Tokenize multiple DCLM shards in parallel (7 jobs for shards 2-8)
# Usage: bash scripts/dataset/tokenize_multiple_shards.sh
#
# Shard 1 is already tokenized from previous run.

# Load site configuration (run from the repository root)
source scripts/config.sh

set -euo pipefail

# 7 shards to tokenize (shard 1 already done)
SHARDS=(
    "03 2"
    "03 3"
    "03 4"
    "03 5"
    "03 6"
    "03 7"
    "03 8"
)

echo "Submitting ${#SHARDS[@]} tokenization jobs for DCLM shards..."
echo ""

for shard in "${SHARDS[@]}"; do
    read -r global local <<< "$shard"
    echo "Submitting: global-shard_${global}/local-shard_${local}"
    GLOBAL_SHARD=$global LOCAL_SHARD=$local sbatch scripts/dataset/tokenize_local.slurm
done

echo ""
echo "All jobs submitted. Use 'squeue -u \$USER' to monitor progress."
