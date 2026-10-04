#!/usr/bin/env python3
"""Compare domain probe results across models.

Usage:
    python scripts/analysis/domain_probe_compare.py \
        --results \
            "Baseline:results/domain-probe/1.7b_seed7/iter_11029" \
            "DRO eta=0.01:results/domain-probe/1.7b-dro-eta0.01/iter_11029" \
        --output-dir results/domain-probe/comparison
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_result(path):
    """Load raw metrics from a domain probe result directory."""
    raw = torch.load(os.path.join(path, 'raw_metrics.pt'), map_location='cpu')
    with open(os.path.join(path, 'summary.json')) as f:
        summary = json.load(f)
    return raw, summary


def compare_models(model_results, output_dir):
    """Generate comparison table and figures."""
    os.makedirs(output_dir, exist_ok=True)

    names = list(model_results.keys())
    group_labels = None

    # Print comparison table
    print("\n" + "=" * 80)
    print("DOMAIN PROBE COMPARISON")
    print("=" * 80)
    header = f"{'Model':<25} {'Avg Max S':>10} {'Avg ΔS':>10} {'Avg ΔQ':>10} {'Avg MI':>10}"
    print(header)
    print("-" * len(header))

    table_rows = []
    for name in names:
        raw, summary = model_results[name]
        row = {
            'model': name,
            'avg_max_selectivity': summary['avg_max_selectivity'],
            'avg_delta_S': summary['avg_delta_S'],
            'avg_delta_Q': summary['avg_delta_Q'],
            'avg_mi': summary['avg_mutual_information'],
        }
        table_rows.append(row)
        print(f"{name:<25} {row['avg_max_selectivity']:>10.4f} {row['avg_delta_S']:>10.4f} "
              f"{row['avg_delta_Q']:>10.4f} {row['avg_mi']:>10.4f}")

        if group_labels is None:
            group_labels = raw.get('group_labels', raw.get('group_names'))

    # Save table as JSON
    with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
        json.dump(table_rows, f, indent=2)

    # Side-by-side selectivity heatmaps (averaged across layers)
    n_models = len(names)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 10))
    if n_models == 1:
        axes = [axes]

    vmin_s = min(model_results[n][0]['selectivity'].mean(dim=0).min().item() for n in names)
    vmax_s = max(model_results[n][0]['selectivity'].mean(dim=0).max().item() for n in names)

    for idx, name in enumerate(names):
        raw, _ = model_results[name]
        S_avg = raw['selectivity'].mean(dim=0).numpy()
        G = S_avg.shape[1]

        im = axes[idx].imshow(S_avg, aspect='auto', cmap='YlOrRd',
                              interpolation='nearest', vmin=vmin_s, vmax=vmax_s)
        axes[idx].set_title(f'{name}\nSelectivity S', fontsize=12)
        axes[idx].set_xlabel('Domain Group')
        axes[idx].set_ylabel('Expert ID')
        axes[idx].set_xticks(range(G))
        axes[idx].set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selectivity_comparison.png'), dpi=150)
    plt.close()

    # Side-by-side quality heatmaps
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 10))
    if n_models == 1:
        axes = [axes]

    vmin_q = min(model_results[n][0]['quality'].mean(dim=0).min().item() for n in names)
    vmax_q = max(model_results[n][0]['quality'].mean(dim=0).max().item() for n in names)

    for idx, name in enumerate(names):
        raw, _ = model_results[name]
        Q_avg = raw['quality'].mean(dim=0).numpy()
        G = Q_avg.shape[1]

        im = axes[idx].imshow(Q_avg, aspect='auto', cmap='YlOrRd_r',
                              interpolation='nearest', vmin=vmin_q, vmax=vmax_q)
        axes[idx].set_title(f'{name}\nNormalized Loss Q (lower=better)', fontsize=12)
        axes[idx].set_xlabel('Domain Group')
        axes[idx].set_ylabel('Expert ID')
        axes[idx].set_xticks(range(G))
        axes[idx].set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_comparison.png'), dpi=150)
    plt.close()

    # Bar chart of aggregate metrics
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics = ['avg_max_selectivity', 'avg_delta_S', 'avg_delta_Q', 'avg_mi']
    titles = ['Avg Max Selectivity', 'Avg Specialization Margin (ΔS)',
              'Avg Competence Advantage (ΔQ)', 'Avg Mutual Information I(E;G)']

    for ax, metric, title in zip(axes, metrics, titles):
        values = [r[metric] for r in table_rows]
        bars = ax.bar(range(n_models), values, color=['#2196F3', '#FF5722', '#4CAF50',
                                                       '#9C27B0', '#FF9800'][:n_models])
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_title(title, fontsize=11)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()

    # MI per layer comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        _, summary = model_results[name]
        mi = summary['mi_per_layer']
        layer_indices = summary['moe_layers']
        ax.plot(layer_indices, mi, marker='o', label=name)

    ax.set_xlabel('MoE Layer Index')
    ax.set_ylabel('Mutual Information I(E;G)')
    ax.set_title('Expert-Group Mutual Information per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mi_per_layer.png'), dpi=150)
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare domain probe results across models')
    parser.add_argument('--results', nargs='+', required=True,
                        help='Model results in "Name:path" format')
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    model_results = {}
    for spec in args.results:
        name, path = spec.split(':', 1)
        raw, summary = load_result(path)
        model_results[name] = (raw, summary)

    compare_models(model_results, args.output_dir)


if __name__ == '__main__':
    main()
