#!/bin/bash
# Oracle single-expert probe driver (290M-32e baseline vs DRMoET pair).
# Modes (env): SELFTEST=true | ROLE=baseline|drmoet ; EVAL_SPEC, POSITIONS.
# Cost: ~32 forwards per sampled position; 1000 pos x 8 eval sets ~= 2.5-3h.

#SBATCH --job-name=oracle-probe
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# Load site configuration (run from the repository root)
source scripts/config.sh

source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=16
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

BASE=${PROJECT_ROOT}
RES="$BASE/analysis/oracle_probe"
mkdir -p "$RES"

BASELINE_CKPT="$BASE/weights/290m-seed3407-32e"
DRMOET_CKPT="$BASE/weights/290m-dro-activation-beta0.999-eta0.001-alpha1.0-norml2_fine_grained_32e"

EVAL_SPEC="${EVAL_SPEC:-$BASE/scripts/analysis/oracle_evalspec.json}"
SELFTEST="${SELFTEST:-false}"
ROLE="${ROLE:-baseline}"
POSITIONS="${POSITIONS:-1000}"

MODEL_ARGS=(
  --num-layers 9 --hidden-size 1024 --ffn-hidden-size 5472 --num-attention-heads 16
  --seq-length 2048 --max-position-embeddings 2048
  --position-embedding-type rope --rotary-percent 1.0
  --normalization RMSNorm --norm-epsilon 1e-6
  --swiglu --untie-embeddings-and-output-weights --disable-bias-linear
  --attention-dropout 0.0 --hidden-dropout 0.0
  --num-experts 32 --moe-router-topk 6
  --moe-ffn-hidden-size 704 --moe-shared-expert-intermediate-size 1408
  --moe-layer-freq "[0]*1+[1]*8"
  --moe-router-pre-softmax --moe-router-dtype fp32
  --moe-router-score-function softmax
  --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.01
  --moe-token-dispatcher-type alltoall
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model EleutherAI/pythia-12b
  --bf16
)
INFRA_ARGS=(
  --pipeline-model-parallel-size 1 --expert-model-parallel-size 1
  --use-distributed-optimizer
)

run_probe() {  # role ckpt extra...
  local role=$1 ckpt=$2; shift 2
  export RDZV_ENDPOINT="localhost:$((29500 + RANDOM % 2000))"
  ( cd "$BASE/Megatron-LM" && torchrun \
      --nnodes 1 --node_rank 0 --nproc_per_node 1 \
      --rdzv-id "${SLURM_JOB_ID}-${role}" --rdzv-backend c10d \
      --rdzv-endpoint "$RDZV_ENDPOINT" \
      ../scripts/analysis/oracle_expert_probe.py \
      --role "$role" \
      --eval-spec "$EVAL_SPEC" \
      --records-out "$RES/records_${role}.jsonl" \
      --positions "$POSITIONS" \
      "$@" \
      "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" \
      --load "$ckpt" --use-checkpoint-args )
}

if [ "$SELFTEST" = "true" ]; then
  echo "=== ORACLE SELF-TEST (baseline checkpoint) ==="
  run_probe baseline "$BASELINE_CKPT" --self-test
  exit 0
fi

if [ "$ROLE" = "baseline" ]; then
  run_probe baseline "$BASELINE_CKPT"
else
  run_probe drmoet "$DRMOET_CKPT"
fi
echo "=== oracle probe ($ROLE) done ==="
