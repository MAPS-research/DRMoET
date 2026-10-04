#!/bin/bash

#SBATCH --job-name=evaluate-all-ckpts
#SBATCH --output=logs/evaluate-local-%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --export=ALL

# Usage:
#   MODEL_SIZE_ID=1.7b sbatch scripts/evaluate_all_checkpoints.slurm

# Load site configuration (run from the repository root)
source scripts/config.sh

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source "$CONDA_ROOT/bin/activate"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# Validate MODEL_SIZE_ID
if [ -z "$MODEL_SIZE_ID" ]; then
    echo "Error: MODEL_SIZE_ID environment variable must be set"
    echo "Usage: MODEL_SIZE_ID=<model_size> sbatch scripts/evaluate_all_checkpoints.slurm"
    exit 1
fi

# Set model-specific variables based on MODEL_SIZE_ID
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
    1.7b*)
        export NUM_LAYERS=18
        export HIDDEN_SIZE=2048
        export FFN_HIDDEN_SIZE=10944
        export MOE_FFN_HIDDEN_SIZE=1408
        export MOE_LAYER_FREQ="[0]*1+[1]*17"
        ;;
    *)
        echo "Error: Could not determine base model size from MODEL_SIZE_ID: $MODEL_SIZE_ID"
        exit 1
        ;;
esac

source configs/model/${MODEL_CONFIG:-flame-moe.sh}

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

CHECKPOINT_DIR="$SSD_WEIGHTS/$MODEL_SIZE_ID"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Find all iteration directories and sort numerically
ALL_ITERS=($(ls -1 "$CHECKPOINT_DIR" | grep -E "^iter_[0-9]+" | sed 's/iter_0*//' | sort -n))
TOTAL_CHECKPOINTS=${#ALL_ITERS[@]}

if [ $TOTAL_CHECKPOINTS -eq 0 ]; then
    echo "Error: No iteration checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo "=========================================="
echo "Model: $MODEL_SIZE_ID"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Total checkpoints: $TOTAL_CHECKPOINTS"
echo "Iterations: ${ALL_ITERS[*]}"
echo "=========================================="

OUTPUT_DIR="${PROJECT_ROOT}/logs/evaluate-local/${MODEL_SIZE_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Create or append to results CSV file
RESULTS_FILE="$OUTPUT_DIR/results_summary.csv"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "iter,arc_challenge,arc_challenge_norm,arc_easy,arc_easy_norm,hellaswag,hellaswag_norm,openbookqa,openbookqa_norm,piqa,piqa_norm,winogrande,sciq,record,truthfulqa_mc2" > "$RESULTS_FILE"
fi

cd lm-evaluation-harness

# Evaluate each checkpoint (skip iterations below START_ITER if set)
for ITER in "${ALL_ITERS[@]}"; do
    if [[ -n "$START_ITER" && "$ITER" -lt "$START_ITER" ]]; then
        echo "Skipping iteration $ITER (below START_ITER=$START_ITER)"
        continue
    fi
    echo ""
    echo "=========================================="
    echo "Evaluating iteration: $ITER"
    echo "=========================================="

    # Set the iteration in the checkpoint file
    echo $ITER > "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"

    LOG_FILE_0SHOT="$OUTPUT_DIR/0shot-iter${ITER}.log"

    # Use dynamic port based on job ID to avoid conflicts
    PORT_0SHOT=$((29500 + (SLURM_JOB_ID % 1000) * 2))

    # Run 0-shot evaluation
    echo "Starting 0-shot evaluation (port $PORT_0SHOT)..."
    CKPT=$ITER PYTHONPATH=$PWD/../Megatron-LM CUDA_VISIBLE_DEVICES=0 \
        torchrun --nproc_per_node=1 --master_port $PORT_0SHOT -m lm_eval \
        ${MODEL_ARGS[@]} \
        --bf16 \
        --seq-length 2048 \
        --micro-batch-size 32 \
        --num_fewshot 0 \
        --batch_size 128 \
        --max-tokens-to-oom 10000000 \
        --seed 42 \
        --load "$CHECKPOINT_DIR" \
        --model megatron_lm \
        --tasks "openbookqa,winogrande,piqa,arc_easy,arc_challenge,hellaswag,sciq,record,truthfulqa_mc2" \
        --output_path "${PROJECT_ROOT}/results/flame-moe/$SLURM_JOB_ID/iter_${ITER}_0shot" > "$LOG_FILE_0SHOT" 2>&1
    echo "Iter $ITER evaluation complete."

    # Extract metrics immediately for this iteration
    # Function to extract a specific metric (acc or acc_norm) for a task
    extract_metric() {
        local log_file="$1"
        local task="$2"
        local metric="$3"
        grep -E "^\|" "$log_file" 2>/dev/null | \
            awk -F'|' -v task="$task" -v metric="$metric" '
            BEGIN { found_task=0 }
            {
                gsub(/^[ \t]+|[ \t]+$/, "", $2)
                gsub(/^[ \t]+|[ \t]+$/, "", $6)
                gsub(/^[ \t]+|[ \t]+$/, "", $8)
                if ($2 == task) {
                    found_task = 1
                    if ($6 == metric) { print $8; exit }
                } else if (found_task && $2 == "" && $6 == metric) {
                    print $8; exit
                } else if (found_task && $2 != "") {
                    found_task = 0
                }
            }'
    }

    arc_c=$(extract_metric "$LOG_FILE_0SHOT" "arc_challenge" "acc")
    arc_c_n=$(extract_metric "$LOG_FILE_0SHOT" "arc_challenge" "acc_norm")
    arc_e=$(extract_metric "$LOG_FILE_0SHOT" "arc_easy" "acc")
    arc_e_n=$(extract_metric "$LOG_FILE_0SHOT" "arc_easy" "acc_norm")
    hella=$(extract_metric "$LOG_FILE_0SHOT" "hellaswag" "acc")
    hella_n=$(extract_metric "$LOG_FILE_0SHOT" "hellaswag" "acc_norm")
    obqa=$(extract_metric "$LOG_FILE_0SHOT" "openbookqa" "acc")
    obqa_n=$(extract_metric "$LOG_FILE_0SHOT" "openbookqa" "acc_norm")
    piqa=$(extract_metric "$LOG_FILE_0SHOT" "piqa" "acc")
    piqa_n=$(extract_metric "$LOG_FILE_0SHOT" "piqa" "acc_norm")
    wino=$(extract_metric "$LOG_FILE_0SHOT" "winogrande" "acc")
    sciq=$(extract_metric "$LOG_FILE_0SHOT" "sciq" "acc")
    record=$(extract_metric "$LOG_FILE_0SHOT" "record" "f1")
    tqa_mc2=$(extract_metric "$LOG_FILE_0SHOT" "truthfulqa_mc2" "acc")

    echo "$ITER,$arc_c,$arc_c_n,$arc_e,$arc_e_n,$hella,$hella_n,$obqa,$obqa_n,$piqa,$piqa_n,$wino,$sciq,$record,$tqa_mc2" >> "$RESULTS_FILE"
    echo "Iter $ITER (0-shot): arc_c_n=$arc_c_n, arc_e_n=$arc_e_n, hella_n=$hella_n, wino=$wino, sciq=$sciq, record=$record, tqa_mc2=$tqa_mc2"
done

echo ""
echo "=========================================="
echo "All checkpoints evaluated!"
echo "=========================================="
echo ""
echo "Final Results Summary:"
column -t -s',' "$RESULTS_FILE"

# Generate checkpoint progression plot
echo ""
echo "Generating checkpoint progression plot..."
python scripts/plot_checkpoint_progression.py "$RESULTS_FILE" --no-show
