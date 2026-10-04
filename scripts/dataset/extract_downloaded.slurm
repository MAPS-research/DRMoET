#!/bin/bash

#SBATCH --job-name=extract_zstd
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# Load site configuration (run from the repository root)
source scripts/config.sh

set -euo pipefail

# Use conda's unzstd
export PATH="$CONDA_ROOT/bin:$PATH"

SSD_DATASET="${PROJECT_ROOT}/dataset"
DOWNLOADS_DIR="$SSD_DATASET/downloads"
TEXTFILES_DIR="$SSD_DATASET/textfiles"

mkdir -p "$TEXTFILES_DIR"

echo "Extracting .zstd files from $DOWNLOADS_DIR to $TEXTFILES_DIR"
echo "Total files to process: $(ls $DOWNLOADS_DIR/*.zstd 2>/dev/null | wc -l)"

# Process files in parallel using xargs
find "$DOWNLOADS_DIR" -name "*.zstd" | xargs -P 16 -I {} bash -c '
    file="{}"
    name=$(basename "$file" .zstd)
    out="${PROJECT_ROOT}/dataset/textfiles/$name"

    if [ -f "$out" ]; then
        echo "Skipping (exists): $name"
    else
        echo "Extracting: $name"
        unzstd -c "$file" > "$out" && rm -f "$file"
    fi
'

echo ""
echo "Done! Files in textfiles/: $(ls $TEXTFILES_DIR | wc -l)"
