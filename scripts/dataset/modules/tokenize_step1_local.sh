#!/bin/bash

# Set up environment variables (same as tokenize_local.slurm)

# Load site configuration (run from the repository root)
source scripts/config.sh

export NFS_MOUNT="${NFS_MOUNT:-${PROJECT_ROOT}/tmp}"
export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_DATASET="$SSD_MOUNT/dataset"
export TOKENIZER="EleutherAI/pythia-12b"

# Set up CUDA environment
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate conda environment once at the script level
source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

tokenize() {
    task=$1

    # Skip if task file is empty (already processed).
    [ ! -s "$task" ] && return 0

    # Read local file path (same for both lines since we're reading from local).
    file=$(sed -n '1p' "$task")

    # Skip if already tokenized (both .bin and .idx must exist and be non-empty).
    local_tokenized_dir=$SSD_DATASET/tokenized/$TOKENIZER
    output_base=$(basename "${file%.jsonl}")_text_document
    bin_file="$local_tokenized_dir/${output_base}.bin"
    idx_file="$local_tokenized_dir/${output_base}.idx"
    if [ -s "$bin_file" ] && [ -s "$idx_file" ]; then
        echo "Skipping (already tokenized): $(basename $file)"
        > "$task"
        return 0
    fi

    # Tokenize the file with Megatron-LM using host conda environment (max 3 attempts).
    for i in {1..3}; do
        echo "Tokenizing $file (Attempt $i of 3)"
        cd ${PROJECT_ROOT}/Megatron-LM && \
        python tools/preprocess_data.py \
            --input $file \
            --output-prefix ${file%.jsonl} \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model $TOKENIZER \
            --append-eod \
            --workers $SLURM_CPUS_PER_TASK > /dev/null 2>&1 && break
        echo "Failed to tokenize $file, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to tokenize $file after 3 attempts." >&2
            return 1
        fi
    done

    # Move tokenized files to local dataset directory (max 3 attempts).
    local_tokenized_dir=$SSD_DATASET/tokenized/$TOKENIZER
    mkdir -p $local_tokenized_dir
    for i in {1..3}; do
        echo "Moving tokenized files to local directory (Attempt $i of 3)"
        mv \
            ${file%.jsonl}_text_document.bin \
            ${file%.jsonl}_text_document.idx \
            $local_tokenized_dir/ > /dev/null 2>&1 && break
        echo "Failed to move tokenized files, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to move tokenized files after 3 attempts." >&2
            return 1
        fi
    done

    # Mark task as completed.
    > "$task"
}

export -f tokenize

# Process task files with file locking to avoid conflicts.
find $NFS_MOUNT -type f -name "*.task" | while read -r line; do
    flock -n "$line" -c "tokenize $line" || true
done 