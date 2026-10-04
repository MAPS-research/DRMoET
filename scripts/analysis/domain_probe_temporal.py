#!/usr/bin/env python3
"""Plot how expert specialization evolves across training checkpoints.

Usage:
    python scripts/analysis/domain_probe_temporal.py \
        --results \
            "Baseline:results/domain-probe/290m-seed3407-32e" \
            "DRO:results/domain-probe/290m-dro-activation-beta0.999-eta0.001-alpha1.0-norml2_fine_grained_32e" \
        --output-dir results/domain-probe/temporal_290m
"""

import argparse
import json
import os
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DOMAIN_GROUPS = ['science', 'physical', 'narrative', 'reading', 'coreference', 'truthfulness']
DOMAIN_LABELS = {
    'science': 'Science / Factual',
    'physical': 'Physical Commonsense',
    'narrative': 'Narrative / Event',
    'reading': 'Reading Comprehension',
    'coreference': 'Coreference',
    'truthfulness': 'Truthfulness',
}


def load_temporal_results(result_dir):
    """Load all checkpoint results from a model's result directory.

    Returns dict: {iteration: summary_dict}, sorted by iteration.
    """
    results = {}
    for iter_dir in glob.glob(os.path.join(result_dir, 'iter_*')):
        summary_path = os.path.join(iter_dir, 'summary.json')
        if not os.path.exists(summary_path):
            continue
        iter_num = int(os.path.basename(iter_dir).replace('iter_', ''))
        with open(summary_path) as f:
            results[iter_num] = json.load(f)

    return dict(sorted(results.items()))


def load_temporal_raw(result_dir):
    """Load raw metrics (selectivity tensors) for all checkpoints."""
    results = {}
    for iter_dir in glob.glob(os.path.join(result_dir, 'iter_*')):
        raw_path = os.path.join(iter_dir, 'raw_metrics.pt')
        if not os.path.exists(raw_path):
            continue
        iter_num = int(os.path.basename(iter_dir).replace('iter_', ''))
        results[iter_num] = torch.load(raw_path, map_location='cpu', weights_only=False)

    return dict(sorted(results.items()))


def plot_mi_heatmap(model_results, output_dir):
    """Plot MI per layer as a heatmap over training iterations, one per model."""
    for model_name, summaries in model_results.items():
        iters = list(summaries.keys())
        if not iters:
            continue

        layers = summaries[iters[0]]['moe_layers']
        mi_matrix = np.array([summaries[it]['mi_per_layer'] for it in iters]).T  # [L, T]

        fig, ax = plt.subplots(figsize=(max(12, len(iters) * 0.4), 6))
        im = ax.imshow(mi_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('MoE Layer')
        ax.set_title(f'Expert-Group Mutual Information I(E;G)\n{model_name}')
        ax.set_xticks(range(len(iters)))
        ax.set_xticklabels([str(it) for it in iters], rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) for l in layers])
        plt.colorbar(im, ax=ax, label='MI')
        plt.tight_layout()
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'mi_heatmap_{safe_name}.png'), dpi=150)
        plt.close()


def plot_aggregate_metrics(model_results, output_dir):
    """Plot aggregate specialization metrics over training iterations."""
    metrics = [
        ('avg_mutual_information', 'Avg Mutual Information I(E;G)'),
        ('avg_delta_S', 'Avg Specialization Margin (ΔS)'),
        ('avg_delta_Q', 'Avg Competence Advantage (ΔQ)'),
        ('avg_max_selectivity', 'Avg Max Selectivity'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    for ax, (key, title) in zip(axes, metrics):
        for idx, (model_name, summaries) in enumerate(model_results.items()):
            iters = list(summaries.keys())
            values = [summaries[it][key] for it in iters]
            ax.plot(iters, values, marker='o', markersize=3, label=model_name,
                    color=colors[idx % len(colors)], linewidth=1.5)
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregate_metrics.png'), dpi=150)
    plt.close()


def plot_per_group_selectivity(model_raw_results, output_dir):
    """Plot mean selectivity per domain group over training iterations."""
    # Determine groups from first available result
    first_model = next(iter(model_raw_results.values()))
    first_iter = next(iter(first_model.values()))
    groups = first_iter.get('group_names', DOMAIN_GROUPS)
    G = len(groups)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    for g_idx in range(min(G, 6)):
        ax = axes[g_idx]
        group_name = groups[g_idx]

        for m_idx, (model_name, raw_data) in enumerate(model_raw_results.items()):
            iters = sorted(raw_data.keys())
            mean_sel = []
            for it in iters:
                S = raw_data[it]['selectivity']  # [L, E, G]
                # Mean selectivity for this group across all experts and layers
                mean_sel.append(S[:, :, g_idx].mean().item())

            ax.plot(iters, mean_sel, marker='o', markersize=3, label=model_name,
                    color=colors[m_idx % len(colors)], linewidth=1.5)

        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Mean Selectivity')
        label = DOMAIN_LABELS.get(group_name, group_name)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_group_selectivity.png'), dpi=150)
    plt.close()


def plot_mi_comparison(model_results, output_dir):
    """Plot avg MI over time for all models on one chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    for idx, (model_name, summaries) in enumerate(model_results.items()):
        iters = list(summaries.keys())
        mi = [summaries[it]['avg_mutual_information'] for it in iters]
        ax.plot(iters, mi, marker='o', markersize=4, label=model_name,
                color=colors[idx % len(colors)], linewidth=2)

    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Avg Mutual Information I(E;G)', fontsize=12)
    ax.set_title('Expert-Group Mutual Information Over Training', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mi_comparison.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot expert specialization over training')
    parser.add_argument('--results', nargs='+', required=True,
                        help='Model results in "Name:path" format')
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_summaries = {}
    model_raw = {}

    for spec in args.results:
        name, path = spec.split(':', 1)
        summaries = load_temporal_results(path)
        raw = load_temporal_raw(path)
        print(f"  {name}: {len(summaries)} checkpoints loaded from {path}")
        if summaries:
            model_summaries[name] = summaries
        if raw:
            model_raw[name] = raw

    if not model_summaries:
        print("ERROR: No results found!")
        return

    # Print summary table
    print("\n=== Temporal Summary ===")
    for name, summaries in model_summaries.items():
        iters = list(summaries.keys())
        first = summaries[iters[0]]
        last = summaries[iters[-1]]
        print(f"\n{name} ({len(iters)} checkpoints, iter {iters[0]}-{iters[-1]}):")
        print(f"  MI:   {first['avg_mutual_information']:.4f} -> {last['avg_mutual_information']:.4f}")
        print(f"  ΔS:   {first['avg_delta_S']:.4f} -> {last['avg_delta_S']:.4f}")
        print(f"  ΔQ:   {first['avg_delta_Q']:.4f} -> {last['avg_delta_Q']:.4f}")
        print(f"  MaxS: {first['avg_max_selectivity']:.4f} -> {last['avg_max_selectivity']:.4f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_mi_heatmap(model_summaries, args.output_dir)
    plot_aggregate_metrics(model_summaries, args.output_dir)
    plot_mi_comparison(model_summaries, args.output_dir)

    if model_raw:
        plot_per_group_selectivity(model_raw, args.output_dir)

    print(f"Plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()
