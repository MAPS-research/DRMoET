#!/bin/bash

#SBATCH --job-name=download_local
#SBATCH --output=logs/%x-%j.log
#SBATCH --time=2-00:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Configurable shard parameters (can be set via environment variables)
# Load site configuration (run from the repository root)
source scripts/config.sh

GLOBAL_SHARD=${GLOBAL_SHARD:-"03"}  # Default to current shard
LOCAL_SHARD=${LOCAL_SHARD:-"1"}      # Default to current shard

# Singularity path
ext3_path="${SINGULARITY_OVERLAY:?set SINGULARITY_OVERLAY in scripts/config.sh}"
sif_path="${SINGULARITY_IMAGE:?set SINGULARITY_IMAGE in scripts/config.sh}"

# Set up environment variables
export NFS_MOUNT="${PROJECT_ROOT}/tmp"
export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_DATASET="$SSD_MOUNT/dataset"
export SSD_WEIGHTS="$SSD_MOUNT/weights"
export DATASET="dclm28b"

# Create necessary directories
mkdir -p $SSD_DATASET/textfiles
mkdir -p $NFS_MOUNT

trap "rm -rf $NFS_MOUNT" EXIT

download_dclm28b() {
    # Each task file contains the S3 link and the local file path.
    prefix=s3://commoncrawl/contrib/datacomp/DCLM-baseline/global-shard_${GLOBAL_SHARD}_of_10/local-shard_${LOCAL_SHARD}_of_10/

    echo "Downloading from: $prefix"

    # Use AWS CLI directly (outside singularity)
    aws s3 ls $prefix | while read -r line; do
        name=$(echo $line | awk '{print $4}')
        link=$prefix$name
        name=gs${GLOBAL_SHARD}10-ls${LOCAL_SHARD}10_$name
        file=$SSD_DATASET/downloads/$name
        task=$NFS_MOUNT/$name.task
        echo $link >> $task
        echo $file >> $task
    done
    # Dispatch the tasks to the nodes.
    srun -W 0 scripts/dataset/modules/download_dclm_step1_local.sh
}

case $DATASET in
    dclm28b)
        download_dclm28b
        ;;
    *)
        echo "Unknown dataset: $1"
        exit 1
esac 