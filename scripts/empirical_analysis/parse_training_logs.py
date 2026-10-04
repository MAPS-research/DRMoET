#!/usr/bin/env python3
"""
Parse Megatron training logs and compute throughput (TFLOPS) statistics.
Useful for comparing training efficiency between different methods (e.g., baseline vs DRO).
"""

import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class IterationMetrics:
    iteration: int
    consumed_samples: int
    elapsed_time_ms: float
    tflops: float
    learning_rate: float
    lm_loss: float
    lm_loss_base: Optional[float]  # Only in DRO logs
    z_loss: float
    load_balancing_loss: float
    grad_norm: float


def parse_log_line(line: str) -> Optional[IterationMetrics]:
    """Parse a single training log line and extract metrics."""
    # Match iteration line pattern
    pattern = r'iteration\s+(\d+)/\s*\d+\s*\|.*consumed samples:\s*(\d+).*elapsed time per iteration \(ms\):\s*([\d.]+).*throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+).*learning rate:\s*([\d.E+-]+).*lm loss:\s*([\d.E+-]+)'

    match = re.search(pattern, line)
    if not match:
        return None

    iteration = int(match.group(1))
    consumed_samples = int(match.group(2))
    elapsed_time_ms = float(match.group(3))
    tflops = float(match.group(4))
    learning_rate = float(match.group(5))
    lm_loss = float(match.group(6))

    # Extract optional lm_loss_base (DRO only)
    lm_loss_base_match = re.search(r'lm loss base:\s*([\d.E+-]+)', line)
    lm_loss_base = float(lm_loss_base_match.group(1)) if lm_loss_base_match else None

    # Extract z_loss
    z_loss_match = re.search(r'z_loss:\s*([\d.E+-]+)', line)
    z_loss = float(z_loss_match.group(1)) if z_loss_match else 0.0

    # Extract load_balancing_loss
    lb_loss_match = re.search(r'load_balancing_loss:\s*([\d.E+-]+)', line)
    lb_loss = float(lb_loss_match.group(1)) if lb_loss_match else 0.0

    # Extract grad_norm
    grad_norm_match = re.search(r'grad norm:\s*([\d.]+)', line)
    grad_norm = float(grad_norm_match.group(1)) if grad_norm_match else 0.0

    return IterationMetrics(
        iteration=iteration,
        consumed_samples=consumed_samples,
        elapsed_time_ms=elapsed_time_ms,
        tflops=tflops,
        learning_rate=learning_rate,
        lm_loss=lm_loss,
        lm_loss_base=lm_loss_base,
        z_loss=z_loss,
        load_balancing_loss=lb_loss,
        grad_norm=grad_norm
    )


def parse_log_file(log_path: Path) -> list[IterationMetrics]:
    """Parse entire log file and return list of metrics."""
    metrics = []
    with open(log_path, 'r') as f:
        for line in f:
            m = parse_log_line(line)
            if m is not None:
                metrics.append(m)
    return metrics


def compute_statistics(values: np.ndarray, skip_warmup: int = 0) -> dict:
    """Compute statistics for a metric, optionally skipping warmup iterations."""
    values = values[skip_warmup:]
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'p5': np.percentile(values, 5),
        'p95': np.percentile(values, 95),
        'count': len(values)
    }


def analyze_log(log_path: Path, name: str, skip_warmup: int = 10) -> dict:
    """Analyze a training log and compute comprehensive statistics."""
    metrics = parse_log_file(log_path)

    if not metrics:
        print(f"Warning: No metrics found in {log_path}")
        return {}

    # Extract arrays
    iterations = np.array([m.iteration for m in metrics])
    tflops = np.array([m.tflops for m in metrics])
    elapsed_ms = np.array([m.elapsed_time_ms for m in metrics])
    lm_loss = np.array([m.lm_loss for m in metrics])
    grad_norm = np.array([m.grad_norm for m in metrics])

    # Check if DRO (has lm_loss_base)
    is_dro = metrics[0].lm_loss_base is not None

    print(f"\n{'='*60}")
    print(f"Analysis for: {name}")
    print(f"Log file: {log_path}")
    print(f"{'='*60}")
    print(f"Total iterations: {len(metrics)}")
    print(f"Iteration range: {iterations[0]} - {iterations[-1]}")
    print(f"Method type: {'DRO' if is_dro else 'Baseline'}")
    print(f"Warmup iterations skipped: {skip_warmup}")

    # TFLOPS statistics
    tflops_stats = compute_statistics(tflops, skip_warmup)
    print(f"\n--- Throughput (TFLOP/s/GPU) ---")
    print(f"  Mean:   {tflops_stats['mean']:.2f}")
    print(f"  Std:    {tflops_stats['std']:.2f}")
    print(f"  Median: {tflops_stats['median']:.2f}")
    print(f"  Min:    {tflops_stats['min']:.2f}")
    print(f"  Max:    {tflops_stats['max']:.2f}")
    print(f"  P5:     {tflops_stats['p5']:.2f}")
    print(f"  P95:    {tflops_stats['p95']:.2f}")

    # Time per iteration
    time_stats = compute_statistics(elapsed_ms, skip_warmup)
    print(f"\n--- Time per Iteration (ms) ---")
    print(f"  Mean:   {time_stats['mean']:.2f}")
    print(f"  Std:    {time_stats['std']:.2f}")
    print(f"  Median: {time_stats['median']:.2f}")

    # Loss at end
    print(f"\n--- Training Loss ---")
    print(f"  Initial (iter {iterations[0]}): {lm_loss[0]:.4f}")
    print(f"  Final (iter {iterations[-1]}):   {lm_loss[-1]:.4f}")
    if is_dro:
        lm_loss_base = np.array([m.lm_loss_base for m in metrics])
        print(f"  Final base loss:   {lm_loss_base[-1]:.4f}")

    return {
        'name': name,
        'is_dro': is_dro,
        'iterations': iterations,
        'tflops': tflops,
        'elapsed_ms': elapsed_ms,
        'lm_loss': lm_loss,
        'tflops_stats': tflops_stats,
        'time_stats': time_stats
    }


def compare_methods(baseline: dict, dro: dict, skip_warmup: int = 10):
    """Compare two training methods and print differences."""
    print(f"\n{'='*60}")
    print("COMPARISON: Baseline vs DRO")
    print(f"{'='*60}")

    b_tflops = baseline['tflops_stats']
    d_tflops = dro['tflops_stats']

    # TFLOPS comparison
    tflops_diff = d_tflops['mean'] - b_tflops['mean']
    tflops_pct = (tflops_diff / b_tflops['mean']) * 100

    print(f"\n--- Throughput Comparison (TFLOP/s/GPU) ---")
    print(f"  Baseline mean:  {b_tflops['mean']:.2f} ± {b_tflops['std']:.2f}")
    print(f"  DRO mean:       {d_tflops['mean']:.2f} ± {d_tflops['std']:.2f}")
    print(f"  Difference:     {tflops_diff:+.2f} ({tflops_pct:+.2f}%)")

    # Time comparison
    b_time = baseline['time_stats']
    d_time = dro['time_stats']
    time_diff = d_time['mean'] - b_time['mean']
    time_pct = (time_diff / b_time['mean']) * 100

    print(f"\n--- Time per Iteration (ms) ---")
    print(f"  Baseline mean:  {b_time['mean']:.2f} ± {b_time['std']:.2f}")
    print(f"  DRO mean:       {d_time['mean']:.2f} ± {d_time['std']:.2f}")
    print(f"  Difference:     {time_diff:+.2f} ({time_pct:+.2f}%)")

    # Effective throughput (accounting for overhead)
    print(f"\n--- Summary for Paper ---")
    print(f"  DRO throughput overhead: {abs(tflops_pct):.2f}% {'slower' if tflops_pct < 0 else 'faster'}")
    print(f"  DRO time overhead:       {abs(time_pct):.2f}% {'longer' if time_pct > 0 else 'shorter'}")


def export_for_plotting(baseline: dict, dro: dict, output_path: Path):
    """Export data for plotting in notebooks."""
    import json

    data = {
        'baseline': {
            'iterations': baseline['iterations'].tolist(),
            'tflops': baseline['tflops'].tolist(),
            'elapsed_ms': baseline['elapsed_ms'].tolist(),
            'lm_loss': baseline['lm_loss'].tolist(),
            'stats': {
                'tflops_mean': baseline['tflops_stats']['mean'],
                'tflops_std': baseline['tflops_stats']['std'],
                'time_mean': baseline['time_stats']['mean'],
                'time_std': baseline['time_stats']['std']
            }
        },
        'dro': {
            'iterations': dro['iterations'].tolist(),
            'tflops': dro['tflops'].tolist(),
            'elapsed_ms': dro['elapsed_ms'].tolist(),
            'lm_loss': dro['lm_loss'].tolist(),
            'stats': {
                'tflops_mean': dro['tflops_stats']['mean'],
                'tflops_std': dro['tflops_stats']['std'],
                'time_mean': dro['time_stats']['mean'],
                'time_std': dro['time_stats']['std']
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nExported data to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Parse and compare Megatron training logs')
    parser.add_argument('--baseline', type=Path, required=True, help='Path to baseline log')
    parser.add_argument('--dro', type=Path, required=True, help='Path to DRO log')
    parser.add_argument('--skip-warmup', type=int, default=10,
                        help='Number of warmup iterations to skip (default: 10)')
    parser.add_argument('--export', type=Path, help='Export data to JSON for plotting')

    args = parser.parse_args()

    # Analyze both logs
    baseline_results = analyze_log(args.baseline, "Baseline (290m)", args.skip_warmup)
    dro_results = analyze_log(args.dro, "DRO", args.skip_warmup)

    # Compare methods
    if baseline_results and dro_results:
        compare_methods(baseline_results, dro_results, args.skip_warmup)

        if args.export:
            export_for_plotting(baseline_results, dro_results, args.export)


if __name__ == '__main__':
    main()
