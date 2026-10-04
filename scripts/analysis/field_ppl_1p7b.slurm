#!/bin/bash
# PPL of the 1.7B checkpoints (baseline vs DRO eta sweep) on DCLM (in-dist)
# and Paloma M2D2-S2ORC (shifted). All checkpoints share architecture
# (L18 h2048 E64 top-6, seed 1234) and sit at matched 32k iters for
# baseline / eta=0.0001 / eta=0.1.
#
# 10.28B total params (1.7B active) in bf16 ~= 21GB weights; batch 8 keeps
# the fp32 logits tensor (~3.3GB) comfortable. Checkpoints were saved with
# EP=4 and are resharded to EP=1 on load by torch_dist.

#SBATCH --job-name=ppl-1p7b
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
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

BASE=${PROJECT_ROOT}
RES="${RES_DIR:-$BASE/analysis/ppl_1p7b}"; mkdir -p "$RES"
BS="${BS:-8}"

# tag:path pairs
MODELS="${MODELS:-baseline:$BASE/weights/1.7b_32k_4x4 \
eta0.1:$BASE/weights/1.7b-dro-activation-beta0.999-eta0.1-alpha1.0-norml2_fine_grained_4x4 \
eta0.0001:$BASE/weights/1.7b-dro-activation-beta0.999-eta0.0001-alpha1.0-norml2_fine_grained_4x4}"

D=$BASE/dataset/tokenized/EleutherAI/pythia-12b
DCLM_FIELDS=""
for i in 0 1 2 3; do
  DCLM_FIELDS="$DCLM_FIELDS dclm:$D/gs0310-ls110_shard_0000000${i}_processed_text_document"
done
FIELDS="${FIELDS:-$DCLM_FIELDS $(cat $BASE/scripts/analysis/all_s2orc_domains.txt)}"

MODEL_ARGS=(
  --num-layers 18 --hidden-size 2048 --ffn-hidden-size 10944 --num-attention-heads 16
  --seq-length 2048 --max-position-embeddings 2048
  --position-embedding-type rope --rotary-percent 1.0
  --normalization RMSNorm --norm-epsilon 1e-6
  --swiglu --untie-embeddings-and-output-weights --disable-bias-linear
  --attention-dropout 0.0 --hidden-dropout 0.0
  --num-experts 64 --moe-router-topk 6
  --moe-ffn-hidden-size 1408 --moe-shared-expert-intermediate-size 2816
  --moe-layer-freq "[0]*1+[1]*17"
  --moe-router-pre-softmax --moe-router-dtype fp32 --moe-router-score-function softmax
  --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.01
  --moe-token-dispatcher-type alltoall
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model EleutherAI/pythia-12b --bf16
)
INFRA_ARGS=(--pipeline-model-parallel-size 1 --expert-model-parallel-size 1 --use-distributed-optimizer)

for spec in $MODELS; do
  tag="${spec%%:*}"; ckpt="${spec#*:}"
  if [ -f "$RES/ppl_${tag}.json" ]; then echo "=== $tag already done, skipping ==="; continue; fi
  echo "=== 1.7B PPL: $tag  ($ckpt) ==="
  export RDZV_ENDPOINT="localhost:$((29500 + RANDOM % 2000))"
  ( cd "$BASE/Megatron-LM" && torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1 \
      --rdzv-id "${SLURM_JOB_ID}-${tag}" --rdzv-backend c10d --rdzv-endpoint "$RDZV_ENDPOINT" \
      ../scripts/analysis/field_ppl.py --role "$tag" --fields $FIELDS \
      --batch-size "$BS" --out "$RES/ppl_${tag}.json" \
      "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" --load "$ckpt" --use-checkpoint-args )
done
echo "=== 1.7B PPL sweep done ==="
