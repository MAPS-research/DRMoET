#!/bin/bash

#SBATCH --job-name=evaluate-local
#SBATCH --output=logs/evaluate-local-%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-2:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --export=ALL

# Load site configuration (run from the repository root)
source scripts/config.sh

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Use unfused attention to avoid Flash Attention issues
# export NVTE_UNFUSED_ATTN=1

source "$CONDA_ROOT/bin/activate"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# Note: scripts/secret.sh doesn't exist, so we skip it

# Set model-specific variables based on MODEL_SIZE_ID (without sourcing release scripts)
case "$MODEL_SIZE_ID" in
    38m*)
        export NUM_LAYERS=9
        export HIDDEN_SIZE=256
        export FFN_HIDDEN_SIZE=1368
        export MOE_FFN_HIDDEN_SIZE=176
        export MOE_LAYER_FREQ="[0]*1+[1]*8"
        ;;
    98m*)
        export NUM_LAYERS=9
        export HIDDEN_SIZE=512
        export FFN_HIDDEN_SIZE=2736
        export MOE_FFN_HIDDEN_SIZE=352
        export MOE_LAYER_FREQ="[0]*1+[1]*8"
        ;;
    115m*)
        export NUM_LAYERS=12
        export HIDDEN_SIZE=512
        export FFN_HIDDEN_SIZE=2736
        export MOE_FFN_HIDDEN_SIZE=352
        export MOE_LAYER_FREQ="[0]*1+[1]*11"
        ;;
    290m*)
        export NUM_LAYERS=9
        export HIDDEN_SIZE=1024
        export FFN_HIDDEN_SIZE=5472
        export MOE_FFN_HIDDEN_SIZE=704
        export MOE_LAYER_FREQ="[0]*1+[1]*8"
        ;;
    *)
        echo "Error: Could not determine base model size from MODEL_SIZE_ID: $MODEL_SIZE_ID"
        echo "Supported prefixes: 38m, 98m, 115m"
        exit 1
        ;;
esac

# Source the model configuration to get MODEL_ARGS (same as training scripts)
source configs/model/flame-moe.sh

echo $ITER > "$SSD_WEIGHTS/$MODEL_SIZE_ID/latest_checkpointed_iteration.txt"

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Create logs directory
mkdir -p logs/evaluate-local/$SLURM_JOB_ID

# ITER and MODEL_SIZE_ID should be passed as environment variables
# Example: ITER=<iteration> MODEL_SIZE_ID=<model_size> sbatch scripts/evaluate_local.slurm
if [ -z "$ITER" ]; then
    echo "Error: ITER environment variable must be set"
    echo "Usage: ITER=<iteration> MODEL_SIZE_ID=<model_size> sbatch scripts/evaluate_local.slurm"
    exit 1
fi

if [ -z "$MODEL_SIZE_ID" ]; then
    echo "Error: MODEL_SIZE_ID environment variable must be set"
    echo "Usage: ITER=<iteration> MODEL_SIZE_ID=<model_size> sbatch scripts/evaluate_local.slurm"
    exit 1
fi

echo "Using ITER: $ITER"
echo "Using MODEL_SIZE_ID: $MODEL_SIZE_ID"

# Set the checkpoint directory (parent directory containing the iter_* folders)
CHECKPOINT_DIR="$SSD_WEIGHTS/$MODEL_SIZE_ID"

echo "Using checkpoint: $CHECKPOINT_DIR"

cd lm-evaluation-harness

num_fewshots=(0 10)
fewshot_tasks=(
    # "openbookqa,winogrande"
    "openbookqa,winogrande,piqa,arc_easy,arc_challenge,hellaswag"
    "openbookqa,winogrande,piqa,arc_easy,arc_challenge,hellaswag"
    # "piqa,arc_easy,arc_challenge,hellaswag"
)

# for each fewshot task and corresponding num_fewshot, run the evaluation
for i in "${!num_fewshots[@]}"; do
    echo "Evaluating ${num_fewshots[$i]} shot for ${fewshot_tasks[$i]}"
    CKPT=$ITER PYTHONPATH=$PWD/../Megatron-LM CUDA_VISIBLE_DEVICES=$i torchrun --nproc_per_node=1 --master_port $((12345 + i)) -m lm_eval \
    ${MODEL_ARGS[@]} \
    --bf16 \
    --seq-length 2048 \
    --micro-batch-size 32 \
    --num_fewshot ${num_fewshots[$i]} \
    --batch_size 16 \
    --max-tokens-to-oom 10000000 \
    --seed 42 \
    --load $CHECKPOINT_DIR \
    --model megatron_lm \
    --tasks "${fewshot_tasks[$i]}" \
    --output_path ../results/flame-moe/$SLURM_JOB_ID > ../logs/evaluate-local/$SLURM_JOB_ID/${num_fewshots[$i]}shot-${i}.log 2>&1 &
done

wait