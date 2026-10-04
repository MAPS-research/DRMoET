# GPU Utilization Profiling Guide for FLAME-MoE

## Metrics to Report in Paper

### 1. Throughput Metrics (from training logs - DONE)
- **TFLOP/s/GPU**: Mean ± std
- **Coefficient of Variation (CV)**: Shows stability
- **Time per iteration**: Mean, P10, P50, P90

### 2. Memory Metrics
Collect with `nvidia-smi` during training:
```bash
# Add to your SLURM script to log every 10 seconds:
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory \
    --format=csv -l 10 > gpu_memory_${SLURM_JOB_ID}.csv &
```

### 3. Expert Load Distribution
Add this to your training script to log expert utilization:
```python
# In Megatron's MoE forward pass, log:
# - tokens_per_expert: histogram of token assignment
# - expert_capacity_utilization: actual/max tokens per expert
# - dropped_tokens: tokens exceeding expert capacity
```

### 4. Communication Overhead (NCCL)
Profile with NCCL debug:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
```

### 5. Detailed GPU Profiling (Nsight Systems)
```bash
# Profile 100 iterations after warmup
nsys profile --stats=true --force-overwrite=true \
    -o profile_baseline \
    python pretrain_gpt.py ... --exit-iteration 110

nsys profile --stats=true --force-overwrite=true \
    -o profile_dro \
    python pretrain_gpt_dro.py ... --exit-iteration 110
```

## Recommended Experiments

### Experiment 1: Expert Utilization Analysis
Track per-expert token counts across iterations to show:
- Baseline may have higher expert imbalance (some experts overloaded)
- DRO may achieve more uniform distribution

### Experiment 2: Scaling Analysis
Run both methods at different scales:
- 290M, 1B, 7B parameters
- 2, 4, 8 nodes
Show throughput overhead scales (or doesn't scale) with model size.

### Experiment 3: Memory Bandwidth Test
Use `nvbandwidth` to measure if DRO affects memory patterns:
```bash
# Before and after training session
nvbandwidth -t host_to_device_memcpy_ce
```

### Experiment 4: Ablation on DRO Components
If DRO has multiple components (reweighting, regularization),
ablate each to identify what causes the speedup.

## Paper-Ready Figures

1. **Box plot**: TFLOPS distribution (baseline vs DRO)
2. **Time series**: Throughput over training iterations
3. **Histogram**: Iteration time distribution
4. **Bar chart**: Mean throughput with error bars (95% CI)
5. **Expert heatmap**: Token distribution across experts over time

## Quick Commands to Generate Stats

```bash
# Run the analysis script
python scripts/empirical_analysis/parse_training_logs.py \
    --baseline logs/analysis_log/290m_output.log \
    --dro logs/analysis_log/dro_output.log \
    --skip-warmup 10 \
    --export logs/analysis_log/throughput_comparison.json
```
