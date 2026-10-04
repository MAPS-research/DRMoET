#!/bin/bash

#SBATCH --job-name=tokenize_local
#SBATCH --output=logs/%x/%j.log
#SBATCH --time=00-04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Configurable shard parameters (can be set via environment variables)
# If not set, processes ALL .jsonl files
# Load site configuration (run from the repository root)
source scripts/config.sh

GLOBAL_SHARD=${GLOBAL_SHARD:-""}
LOCAL_SHARD=${LOCAL_SHARD:-""}

# Singularity path
ext3_path="${SINGULARITY_OVERLAY:?set SINGULARITY_OVERLAY in scripts/config.sh}"
sif_path="${SINGULARITY_IMAGE:?set SINGULARITY_IMAGE in scripts/config.sh}"

# Set up environment variables
export NFS_MOUNT="${PROJECT_ROOT}/tmp_${SLURM_JOB_ID}"
export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_DATASET="$SSD_MOUNT/dataset"
export SSD_WEIGHTS="$SSD_MOUNT/weights"
export TOKENIZER="EleutherAI/pythia-12b"

# Create necessary directories
mkdir -p $SSD_DATASET/tokenized/$TOKENIZER
mkdir -p $NFS_MOUNT

# Ensure cleanup of temporary directories on exit.
trap "rm -rf $NFS_MOUNT" EXIT

# Grab files to be tokenized from local directory and store into the parameter queue.
# If GLOBAL_SHARD and LOCAL_SHARD are set, only process files matching that shard.
if [ -n "$GLOBAL_SHARD" ] && [ -n "$LOCAL_SHARD" ]; then
    SHARD_PREFIX="gs${GLOBAL_SHARD}10-ls${LOCAL_SHARD}10_"
    echo "Processing shard: $SHARD_PREFIX"
    find $SSD_DATASET/textfiles -type f -name "${SHARD_PREFIX}*.jsonl" | while read -r line; do
        file=$line
        name=$(basename $line)
        task=$NFS_MOUNT/$name.task
        echo $file >> $task
        echo $file >> $task
    done
else
    echo "Processing ALL .jsonl files"
    find $SSD_DATASET/textfiles -type f -name "*.jsonl" | while read -r line; do
        file=$line
        name=$(basename $line)
        task=$NFS_MOUNT/$name.task
        echo $file >> $task
        echo $file >> $task
    done
fi

# Dispatch the tokenization.
srun -W 0 scripts/dataset/modules/tokenize_step1_local.sh 