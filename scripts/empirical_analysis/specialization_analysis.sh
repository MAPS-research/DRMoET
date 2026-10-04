#!/bin/bash
# Expert Specialization Analysis Pipeline
#
# This script:
# 1. Downloads domain-specific datasets (code, math, wiki, etc.)
# 2. Tokenizes them for Megatron
# 3. Captures expert activations for each domain
# 4. The notebook then analyzes which experts specialize in which domains
#
# Usage:
#   sbatch specialization_analysis.sh
#
# Or run steps individually:
#   STEP=download sbatch specialization_analysis.sh
#   STEP=tokenize sbatch specialization_analysis.sh
#   STEP=capture sbatch specialization_analysis.sh

#SBATCH --job-name=specialization-analysis
#SBATCH --output=logs/%x/%j.log
#SBATCH --time=0-8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load site configuration (run from the repository root)
source scripts/config.sh

set -euo pipefail

# Set up environment
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

# Configuration
export DOMAINS=${DOMAINS:-"code math wiki web"}
export SAMPLES_PER_DOMAIN=${SAMPLES_PER_DOMAIN:-5000}
export DOMAIN_DATA_DIR=${DOMAIN_DATA_DIR:-data/domain_eval}
export DOMAIN_TOKENIZED_DIR=${DOMAIN_TOKENIZED_DIR:-$SSD_DATASET/domain_eval}
export OUTPUT_DIR=${OUTPUT_DIR:-${SSD_MOUNT}/actives/domain}
export STEP=${STEP:-all}

# Model configs (32-expert models)
export MODEL_NAMES=${MODEL_NAMES:-"290m-seed3407-32e 290m-dro-activation-beta0.999-eta0.001-alpha1.0-norml2_fine_grained_32e"}
export CHECKPOINT=${CHECKPOINT:-16000}
export NUM_EXPERTS=${NUM_EXPERTS:-32}

echo "============================================"
echo "Expert Specialization Analysis Pipeline"
echo "============================================"
echo "Domains: $DOMAINS"
echo "Samples per domain: $SAMPLES_PER_DOMAIN"
echo "Models: $MODEL_NAMES"
echo "Checkpoint: $CHECKPOINT"
echo "Step: $STEP"
echo "============================================"

mkdir -p logs/specialization-analysis
mkdir -p "$DOMAIN_DATA_DIR"

# Step 1: Download domain data
if [ "$STEP" = "all" ] || [ "$STEP" = "download" ]; then
    echo ""
    echo ">>> Step 1: Downloading domain data..."
    python3 scripts/empirical_analysis/prepare_domain_data.py \
        --output-dir "$DOMAIN_DATA_DIR" \
        --samples-per-domain "$SAMPLES_PER_DOMAIN" \
        --domains $DOMAINS
fi

# Step 2: Tokenize domain data
if [ "$STEP" = "all" ] || [ "$STEP" = "tokenize" ]; then
    echo ""
    echo ">>> Step 2: Tokenizing domain data..."

    TOKENIZER="EleutherAI/pythia-12b"

    for domain in $DOMAINS; do
        DOMAIN_JSONL="$DOMAIN_DATA_DIR/$domain.jsonl"
        DOMAIN_OUTPUT="$DOMAIN_TOKENIZED_DIR/$domain"

        if [ ! -f "$DOMAIN_JSONL" ]; then
            echo "  Skipping $domain (no data file)"
            continue
        fi

        if [ -f "$DOMAIN_OUTPUT/${domain}_text_document.bin" ]; then
            echo "  Skipping $domain (already tokenized)"
            continue
        fi

        echo "  Tokenizing $domain..."
        mkdir -p "$DOMAIN_OUTPUT"

        python3 Megatron-LM/tools/preprocess_data.py \
            --input "$DOMAIN_JSONL" \
            --output-prefix "$DOMAIN_OUTPUT/${domain}_text_document" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER" \
            --json-keys text \
            --workers 16 \
            --append-eod
    done
fi

# Step 3: Capture activations for each domain
if [ "$STEP" = "all" ] || [ "$STEP" = "capture" ]; then
    echo ""
    echo ">>> Step 3: Capturing expert activations..."

    # Model architecture for FLAME-MoE-290M-32e
    export NUM_LAYERS=9
    export HIDDEN_SIZE=1024
    export FFN_HIDDEN_SIZE=5472
    export MOE_FFN_HIDDEN_SIZE=704
    export MOE_LAYER_FREQ="[0]*1+[1]*8"
    export MOE_ROUTER_TOPK=6
    export MICRO_BATCH_SIZE=4
    export PIPELINE_MODEL_PARALLEL_SIZE=1
    export EXPERT_MODEL_PARALLEL_SIZE=2
    export RDZV_BACKEND="c10d"

    PORT=29500
    for model_name in $MODEL_NAMES; do
        echo ""
        echo "  Model: $model_name"
        export TRAIN_WEIGHTS=$SSD_WEIGHTS/$model_name

        for domain in $DOMAINS; do
            DOMAIN_TOKENIZED="$DOMAIN_TOKENIZED_DIR/$domain"

            if [ ! -f "$DOMAIN_TOKENIZED/${domain}_text_document.bin" ]; then
                echo "    Skipping $domain (not tokenized)"
                continue
            fi

            echo "    Capturing $domain on port $PORT..."
            export RDZV_ENDPOINT="localhost:$PORT"
            export EACT_SAVE="$OUTPUT_DIR/$model_name/$CHECKPOINT/$domain"
            export TRAIN_DATASET="$DOMAIN_TOKENIZED"

            # Set checkpoint
            echo $CHECKPOINT > $TRAIN_WEIGHTS/latest_checkpointed_iteration.txt

            mkdir -p "$EACT_SAVE"

            # Run capture (using existing capture infrastructure)
            scripts/empirical_analysis/modules/capture_step1.sh
            scripts/empirical_analysis/modules/capture_step2.sh

            PORT=$((PORT + 1))
            sleep 3
        done
    done
fi

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "Actives saved to: $OUTPUT_DIR"
echo ""
echo "Next: Run the specialization analysis notebook"
echo "  scripts/empirical_analysis/expert_specialization.ipynb"
echo "============================================"
