#!/bin/bash
# Download and extract the DCLM dataset from S3, save locally.
# Invoked by scripts/dataset/download_local.slurm

# Author: Hao Kang
# Date: March 9, 2025

# Singularity path

# Load site configuration (run from the repository root)
source scripts/config.sh

ext3_path="${SINGULARITY_OVERLAY:?set SINGULARITY_OVERLAY in scripts/config.sh}"
sif_path="${SINGULARITY_IMAGE:?set SINGULARITY_IMAGE in scripts/config.sh}"

# Set up environment variables (same as download_local.slurm)
export NFS_MOUNT="${PROJECT_ROOT}/tmp"
export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_DATASET="$SSD_MOUNT/dataset"

download() {
    task=$1

    # Skip if task file is empty (already processed)
    [ ! -s "$task" ] && return 0

    # Read S3 link and local file path
    link=$(sed -n '1p' "$task")
    file=$(sed -n '2p' "$task")

    # Download from S3 (max 3 attempts)
    for i in {1..3}; do
        echo "Downloading $link (Attempt $i of 3)"
        aws s3 cp $link $file > /dev/null 2>&1 && break
        echo "Failed to download $link, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to download $link after 3 attempts." >&2
            return 1
        fi
    done

    # Check if download was successful
    if [ ! -f "$file" ]; then
        echo "ERROR: Download completed but file $file does not exist." >&2
        return 1
    fi

    # Extract .zstd file (max 3 attempts)
    for i in {1..3}; do
        echo "Extracting $file (Attempt $i of 3)"
        $CONDA_ROOT/bin/unzstd -f $file && rm -f $file && break
        echo "Failed to extract $file, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to extract $file after 3 attempts." >&2
            return 1
        fi
    done

    # Move extracted file to local dataset directory (max 3 attempts)
    file=${file%.zstd}
    local_dataset_dir=$SSD_DATASET/textfiles
    mkdir -p $local_dataset_dir
    for i in {1..3}; do
        echo "Moving $file to local dataset directory (Attempt $i of 3)"
        mv $file $local_dataset_dir/ > /dev/null 2>&1 && break
        echo "Failed to move $file, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to move $file after 3 attempts." >&2
            return 1
        fi
    done

    # Mark task as completed
    > "$task"
}

export -f download

# Process task files with file locking to avoid conflicts.
find $NFS_MOUNT -type f -name "*.task" | while read -r line; do
    flock -n "$line" -c "download $line" || true
done 