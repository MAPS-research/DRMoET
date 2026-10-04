#!/bin/bash

#SBATCH --job-name=misroute-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --export=ALL

# Usage:
#   MODEL_SIZE_ID=290m-seed3407-32e ITER=16000 sbatch scripts/analysis/misroute_probe.slurm

# Load site configuration (run from the repository root)
source scripts/config.sh

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64

source "$CONDA_ROOT/bin/activate"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export OMP_NUM_THREADS=8
export MEGATRON_USE_TORCH_COMPILE=1

if [ -z "${MODEL_SIZE_ID:-}" ]; then
    echo "Error: MODEL_SIZE_ID must be set"
    exit 1
fi

# Set model architecture based on MODEL_SIZE_ID
case "$MODEL_SIZE_ID" in
    290m*)
        NUM_LAYERS=9; HIDDEN_SIZE=1024; FFN_HIDDEN_SIZE=5472
        MOE_FFN_HIDDEN_SIZE=704; MOE_LAYER_FREQ="[0]*1+[1]*8"
        NUM_EXPERTS=32; NUM_ATTENTION_HEADS=16
        ;;
    1.7b*)
        NUM_LAYERS=18; HIDDEN_SIZE=2048; FFN_HIDDEN_SIZE=10944
        MOE_FFN_HIDDEN_SIZE=1408; MOE_LAYER_FREQ="[0]*1+[1]*17"
        NUM_EXPERTS=64; NUM_ATTENTION_HEADS=16
        ;;
    *)
        echo "Error: Unknown MODEL_SIZE_ID: $MODEL_SIZE_ID"
        exit 1
        ;;
esac

CHECKPOINT_DIR="${PROJECT_ROOT}/weights/$MODEL_SIZE_ID"
OUTPUT_DIR="${PROJECT_ROOT}/results/misroute-probe/${MODEL_SIZE_ID}"

if [ -n "${ITER:-}" ]; then
    echo "$ITER" > "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
    OUTPUT_DIR="${OUTPUT_DIR}/iter_${ITER}"
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Misroute Probe: $MODEL_SIZE_ID"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

PYTHONPATH=$PWD/Megatron-LM:$PWD/lm-evaluation-harness \
torchrun --nproc_per_node=1 --master_port $((29500 + SLURM_JOB_ID % 1000)) \
    scripts/analysis/misroute_probe.py \
    --load "$CHECKPOINT_DIR" \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE \
    --moe-layer-freq "$MOE_LAYER_FREQ" \
    --num-experts $NUM_EXPERTS \
    --moe-router-topk 6 \
    --moe-shared-expert-intermediate-size $((2 * MOE_FFN_HIDDEN_SIZE)) \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --position-embedding-type rope \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model EleutherAI/pythia-12b \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --bf16 \
    --micro-batch-size 1 \
    --moe-router-pre-softmax \
    --moe-router-score-function softmax \
    --moe-router-dtype fp32 \
    --output-dir "$OUTPUT_DIR" \
    --examples-per-group ${EXAMPLES_PER_GROUP:-100000} \
    ${USE_TRAINING_DATA:+--use-training-data} \
    --num-sequences ${NUM_SEQUENCES:-10000}

echo "Misroute probe complete."
