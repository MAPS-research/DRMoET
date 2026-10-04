#!/bin/bash
# Routing-boundary failure probe driver (290M-32e baseline vs DRMoET pair).
#
# Modes (env):
#   SELFTEST=true             -> run correctness self-tests only (baseline ckpt)
#   ROLE=baseline|drmoet|both -> probe roles to run (default both, baseline first
#                                since drmoet consumes its positions file)
#   EVAL_SPEC=<json path>     -> eval-set spec (default: full spec)
#
# One GPU; ~3-5 min per (role, eval set) at 512 test seqs / 8 samples per seq.

#SBATCH --job-name=routing-boundary
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=0-04:00:00
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
# determinism: required for the boundary probe's counterfactual comparisons
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

BASE=${PROJECT_ROOT}
RES="${RES_DIR:-$BASE/analysis/routing_boundary}"
mkdir -p "$RES"

BASELINE_CKPT="${BASELINE_CKPT:-$BASE/weights/290m-seed3407-32e}"
DRMOET_CKPT="${DRMOET_CKPT:-$BASE/weights/290m-dro-activation-beta0.999-eta0.001-alpha1.0-norml2_fine_grained_32e}"

EVAL_SPEC="${EVAL_SPEC:-$BASE/scripts/analysis/routing_boundary_evalspec.json}"
SELFTEST="${SELFTEST:-false}"
ROLE="${ROLE:-both}"

# Arch args mirror the CHECKPOINT args exactly (read from common.pt):
# 9 layers (1 dense + 8 MoE), h1024, ffn5472, 16 heads, 32 experts top-6,
# moe_ffn 704, shared 1408, pre-softmax softmax routing, no scaling factor.
# Arch defaults = 290M pair; override via env for the 98M replication pair.
MODEL_ARGS=(
  --num-layers 9 --hidden-size "${HIDDEN_SIZE:-1024}"
  --ffn-hidden-size "${FFN_HIDDEN_SIZE:-5472}" --num-attention-heads 16
  --seq-length 2048 --max-position-embeddings 2048
  --position-embedding-type rope --rotary-percent 1.0
  --normalization RMSNorm --norm-epsilon 1e-6
  --swiglu --untie-embeddings-and-output-weights --disable-bias-linear
  --attention-dropout 0.0 --hidden-dropout 0.0
  --num-experts 32 --moe-router-topk 6
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE:-704}"
  --moe-shared-expert-intermediate-size "${MOE_SHARED:-1408}"
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
      ../scripts/analysis/routing_boundary_probe.py \
      --role "$role" \
      --eval-spec "$EVAL_SPEC" \
      --positions-file "$RES/positions.json" \
      --records-out "$RES/records_${role}.jsonl" \
      --samples-per-seq "${SAMPLES_PER_SEQ:-8}" \
      "$@" \
      "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" \
      --load "$ckpt" --use-checkpoint-args )
}

if [ "$SELFTEST" = "true" ]; then
  echo "=== SELF-TEST (baseline checkpoint) ==="
  run_probe baseline "$BASELINE_CKPT" --self-test
  exit 0
fi

if [ "$ROLE" = "baseline" ] || [ "$ROLE" = "both" ]; then
  echo "=== ROLE: baseline ==="
  run_probe baseline "$BASELINE_CKPT"
fi
if [ "$ROLE" = "drmoet" ] || [ "$ROLE" = "both" ]; then
  echo "=== ROLE: drmoet ==="
  run_probe drmoet "$DRMOET_CKPT"
fi

echo "=== probe done; summarize with scripts/analysis/summarize_boundary.py ==="
