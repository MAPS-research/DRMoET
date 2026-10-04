#!/bin/bash
# Download multiple DCLM shards (7 new + 1 existing = 8x data)
# Usage: bash scripts/dataset/download_multiple_shards.sh
#
# This script submits 7 SLURM jobs for shards 2-8.
# Shard 1 is already downloaded and tokenized.

# Load site configuration (run from the repository root)
source scripts/config.sh

set -euo pipefail

# 7 NEW shards from same global shard (03) with sequential local shards
# Shard 1 already downloaded and tokenized - skipping it
SHARDS=(
    "03 2"
    "03 3"
    "03 4"
    "03 5"
    "03 6"
    "03 7"
    "03 8"
)

echo "Submitting ${#SHARDS[@]} download jobs for DCLM shards..."
echo ""

for shard in "${SHARDS[@]}"; do
    read -r global local <<< "$shard"
    echo "Submitting: global-shard_${global}_of_10/local-shard_${local}_of_10"
    GLOBAL_SHARD=$global LOCAL_SHARD=$local sbatch scripts/dataset/download_local.slurm
done

echo ""
echo "All jobs submitted. Use 'squeue -u \$USER' to monitor progress."
