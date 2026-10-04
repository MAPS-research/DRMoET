#!/bin/bash
#SBATCH --job-name=perwin --output=slurm_%j.out --error=slurm_%j.err
#SBATCH --time=0-01:00:00 --nodes=1 --ntasks-per-node=1 --mem=64G --cpus-per-task=16
#SBATCH --gres=gpu:1

# Load site configuration (run from the repository root)
source scripts/config.sh

source "$CONDA_ROOT/bin/activate"; conda activate "$CONDA_ENV"
set -euo pipefail
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
B=${PROJECT_ROOT}; R=$B/analysis/perwindow; mkdir -p $R
MA=(--num-layers 9 --hidden-size 1024 --ffn-hidden-size 5472 --num-attention-heads 16
 --seq-length 2048 --max-position-embeddings 2048 --position-embedding-type rope --rotary-percent 1.0
 --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --untie-embeddings-and-output-weights
 --disable-bias-linear --attention-dropout 0.0 --hidden-dropout 0.0 --num-experts 32 --moe-router-topk 6
 --moe-ffn-hidden-size 704 --moe-shared-expert-intermediate-size 1408 --moe-layer-freq "[0]*1+[1]*8"
 --moe-router-pre-softmax --moe-router-dtype fp32 --moe-router-score-function softmax
 --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.01 --moe-token-dispatcher-type alltoall
 --tokenizer-type HuggingFaceTokenizer --tokenizer-model EleutherAI/pythia-12b --bf16)
IA=(--pipeline-model-parallel-size 1 --expert-model-parallel-size 1 --use-distributed-optimizer)
for r in baseline drmoet; do
  [ $r = baseline ] && C=$B/weights/290m-seed3407-32e || C=$B/weights/290m-dro-activation-beta0.999-eta0.001-alpha1.0-norml2_fine_grained_32e
  ( cd $B/Megatron-LM && torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1 \
     --rdzv-id $SLURM_JOB_ID-$r --rdzv-backend c10d --rdzv-endpoint localhost:$((29500+RANDOM%2000)) \
     ../scripts/analysis/perwindow_ppl.py --role $r --fields stat.ME q-bio.GN q-fin.PR cs.SE \
     --out $R/win_$r.json "${MA[@]}" "${IA[@]}" --load $C --use-checkpoint-args )
done
