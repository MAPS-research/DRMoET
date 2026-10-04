#!/usr/bin/env python3
"""
Compute expert utilization statistics from captured routing traces.
Reuses the same trace files as coactivation analysis.

Usage:
    python expert_utilization_step2.py \
        --actives-path /path/to/actives/model/step/layer \
        --results-path /path/to/results/utilization.json
"""

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from tqdm import tqdm

num_experts = int(os.environ.get("NUM_EXPERTS", 64))
max_workers = 196


def work(actives_file: Path) -> np.ndarray:
    """Process a single trace file and return token counts per expert."""
    values, indices = torch.load(actives_file, map_location="cpu", weights_only=True)
    counts = np.zeros(num_experts, dtype=np.float64)
    for idx in indices.flatten().numpy():
        if idx < num_experts:
            counts[idx] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actives-path", type=str, required=True)
    parser.add_argument("--results-path", type=str, required=True)
    parsed = parser.parse_args()

    print("actives_path:", parsed.actives_path)
    print("results_path:", parsed.results_path)

    actives_path = Path(parsed.actives_path)
    assert actives_path.is_dir()
    results_path = Path(parsed.results_path)
    if results_path.is_file():
        print("Results already exist, skipping.")
        return

    results_path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.zeros(num_experts, dtype=np.float64)

    # Process all .pt files
    pt_files = sorted(actives_path.glob("*.pt"))
    if not pt_files:
        print("No .pt files found!")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        jobs = [executor.submit(work, f) for f in pt_files]
        for future in tqdm(as_completed(jobs), total=len(jobs), desc="Processing", ncols=80, mininterval=5):
            counts += future.result()

    # Compute statistics
    total = counts.sum()
    dist = counts / (total + 1e-10)
    cv = np.std(dist) / np.mean(dist) * 100 if np.mean(dist) > 0 else 0
    sorted_c = np.sort(counts)
    n = len(counts)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_c))) / (n * total) - (n + 1) / n if total > 0 else 0
    entropy = -np.sum(dist * np.log(dist + 1e-10))

    result = {
        "counts": counts.tolist(),
        "total_tokens": float(total),
        "cv_percent": float(cv),
        "gini": float(gini),
        "entropy": float(entropy),
        "max_entropy": float(np.log(num_experts)),
        "max_min_ratio": float(counts.max() / (counts.min() + 1e-10)),
    }

    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"CV: {cv:.2f}%, Gini: {gini:.4f}, Entropy: {entropy:.3f}")


if __name__ == "__main__":
    main()
