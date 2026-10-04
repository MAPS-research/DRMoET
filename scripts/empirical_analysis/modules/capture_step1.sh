#!/bin/bash
# Local version - weights and dataset already exist locally, no download needed.

echo "Using local weights at: $TRAIN_WEIGHTS"
echo "Using local dataset at: $TRAIN_DATASET"

# Verify paths exist
if [ ! -d "$TRAIN_WEIGHTS" ]; then
    echo "ERROR: TRAIN_WEIGHTS directory not found: $TRAIN_WEIGHTS"
    exit 1
fi

if [ ! -d "$SSD_DATASET" ]; then
    echo "ERROR: SSD_DATASET directory not found: $SSD_DATASET"
    exit 1
fi
