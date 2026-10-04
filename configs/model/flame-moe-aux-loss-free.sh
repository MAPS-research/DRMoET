#!/bin/bash
# Model configuration with FLAME MoE (auxiliary-loss-free load balancing).
# Uses per-expert bias instead of auxiliary losses (arXiv 2408.15664).

MODEL_ARGS=(
    # Network Size
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-layers $NUM_LAYERS
    --num-attention-heads 16
    --swiglu
    --max-position-embeddings 2048
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --disable-bias-linear

    # Mixture of Experts (auxiliary-loss-free)
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE
    --num-experts ${NUM_EXPERTS:-32}
    --moe-router-topk ${MOE_ROUTER_TOPK:-6}
    --moe-shared-expert-intermediate-size $((2 * MOE_FFN_HIDDEN_SIZE))
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-router-dtype fp32
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type none
    --moe-router-bias-update-rate 1e-2
    --moe-aux-loss-coeff 0.0
    --moe-z-loss-coeff 0.0

    # Regularization
    --hidden-dropout 0.0
    --attention-dropout 0.0

    # Initialization
    --init-method-std 0.02

    # Tokenizer
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model EleutherAI/pythia-12b
)
