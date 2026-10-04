#!/usr/bin/env python3
"""Expert quality analysis: DRO vs Baseline comparison.

Three experiments:
  1. Router Quality -- entropy, load balance, routing concentration
  2. Per-Expert Domain Quality -- expert loss on hard vs easy domains
  3. Mid-Tier Expert Analysis -- quality by expert traffic tier

Usage:
    python scripts/analysis/expert_quality_analysis.py \
        --baseline results/domain-probe/290m-seed3407-32e \
        --dro results/domain-probe/290m-dro-..._32e \
        --iter 16000 \
        --output-dir results/expert-quality-analysis/290m
"""

import argparse
import json
import os
import glob
import re

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPS = 1e-8
COLOR_BASELINE = '#2196F3'
COLOR_DRO = '#FF5722'

DOMAIN_GROUPS = {
    'science': 'Science / Factual',
    'physical': 'Physical Commonsense',
    'narrative': 'Narrative / Event',
    'reading': 'Reading Comprehension',
    'coreference': 'Coreference / Commonsense',
    'truthfulness': 'Truthfulness',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_metrics(result_dir, iteration):
    """Load raw_metrics.pt for a specific iteration."""
    path = os.path.join(result_dir, f'iter_{iteration}', 'raw_metrics.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No raw_metrics.pt at {path}")
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data


def discover_iterations(result_dir):
    """Discover available iterations in a result directory."""
    iters = []
    for d in glob.glob(os.path.join(result_dir, 'iter_*')):
        if os.path.exists(os.path.join(d, 'raw_metrics.pt')):
            m = re.search(r'iter_(\d+)', d)
            if m:
                iters.append(int(m.group(1)))
    return sorted(iters)


# ---------------------------------------------------------------------------
# Experiment 1: Router Quality
# ---------------------------------------------------------------------------

def compute_routing_entropy(routing_mass):
    """Per-layer routing entropy from routing_mass [L, E, G].
    Returns normalized entropy [L] in [0, 1]."""
    load = routing_mass.sum(dim=2)  # [L, E]
    p = load / (load.sum(dim=1, keepdim=True) + EPS)
    entropy = -(p * torch.log(p + EPS)).sum(dim=1)  # [L]
    E = routing_mass.shape[1]
    max_entropy = torch.log(torch.tensor(E, dtype=torch.float))
    return (entropy / max_entropy).numpy()


def compute_load_imbalance(routing_mass):
    """Load imbalance ratio per layer [L]."""
    load = routing_mass.sum(dim=2)  # [L, E]
    max_load = load.max(dim=1).values
    min_load = load.min(dim=1).values
    return (max_load / (min_load + EPS)).numpy()


def compute_cv(routing_mass):
    """Coefficient of variation of expert load per layer [L]."""
    load = routing_mass.sum(dim=2)  # [L, E]
    return (load.std(dim=1) / (load.mean(dim=1) + EPS)).numpy()


def compute_gini(routing_mass):
    """Gini coefficient of expert load per layer [L]."""
    load = routing_mass.sum(dim=2)  # [L, E]
    L, E = load.shape
    gini = np.zeros(L)
    for l in range(L):
        sorted_load, _ = load[l].sort()
        index = torch.arange(1, E + 1).float()
        gini[l] = (2 * (index * sorted_load).sum() / (E * sorted_load.sum() + EPS) - (E + 1) / E).item()
    return gini


def experiment1(baseline_data, dro_data):
    """Router quality comparison."""
    b_mass = baseline_data['routing_mass']
    d_mass = dro_data['routing_mass']

    results = {
        'baseline_entropy': compute_routing_entropy(b_mass).tolist(),
        'dro_entropy': compute_routing_entropy(d_mass).tolist(),
        'baseline_imbalance': compute_load_imbalance(b_mass).tolist(),
        'dro_imbalance': compute_load_imbalance(d_mass).tolist(),
        'baseline_cv': compute_cv(b_mass).tolist(),
        'dro_cv': compute_cv(d_mass).tolist(),
        'baseline_gini': compute_gini(b_mass).tolist(),
        'dro_gini': compute_gini(d_mass).tolist(),
    }
    # Averages
    for key in ['entropy', 'imbalance', 'cv', 'gini']:
        results[f'baseline_mean_{key}'] = float(np.mean(results[f'baseline_{key}']))
        results[f'dro_mean_{key}'] = float(np.mean(results[f'dro_{key}']))
    return results


def plot_experiment1(results, layer_indices, output_dir):
    """Generate experiment 1 plots."""
    L = len(layer_indices)
    x = np.arange(L)
    width = 0.35

    # Plot 1: Entropy + CV side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Entropy
    ax = axes[0]
    ax.bar(x - width/2, results['baseline_entropy'], width,
           label='Baseline', color=COLOR_BASELINE)
    ax.bar(x + width/2, results['dro_entropy'], width,
           label='DRMoET', color=COLOR_DRO)
    ax.set_xlabel('MoE Layer')
    ax.set_ylabel('Normalized Routing Entropy')
    ax.set_title('Routing Entropy per Layer')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in layer_indices], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    b_mean = results['baseline_mean_entropy']
    d_mean = results['dro_mean_entropy']
    ax.axhline(y=b_mean, color=COLOR_BASELINE, linestyle='--', alpha=0.5)
    ax.axhline(y=d_mean, color=COLOR_DRO, linestyle='--', alpha=0.5)

    # CV + Gini
    ax = axes[1]
    x2 = np.arange(L)
    ax.bar(x2 - width/2, results['baseline_cv'], width,
           label='Baseline CV', color=COLOR_BASELINE)
    ax.bar(x2 + width/2, results['dro_cv'], width,
           label='DRMoET CV', color=COLOR_DRO)
    ax.set_xlabel('MoE Layer')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Load Balance (CV) per Layer')
    ax.set_xticks(x2)
    ax.set_xticklabels([str(i) for i in layer_indices], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(output_dir, f'experiment1_router_quality.{fmt}'),
                    dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 2: Per-Expert Domain Quality
# ---------------------------------------------------------------------------

def compute_domain_loss(quality, routing_mass):
    """Per-domain weighted average loss [G]."""
    expert_loss_mass = quality * routing_mass
    domain_loss = expert_loss_mass.sum(dim=(0, 1)) / (routing_mass.sum(dim=(0, 1)) + EPS)
    return domain_loss.numpy()


def experiment2(baseline_data, dro_data):
    """Per-expert domain quality comparison."""
    b_q = baseline_data['quality']
    b_m = baseline_data['routing_mass']
    d_q = dro_data['quality']
    d_m = dro_data['routing_mass']
    group_names = baseline_data['group_names']

    b_domain_loss = compute_domain_loss(b_q, b_m)
    d_domain_loss = compute_domain_loss(d_q, d_m)
    improvement_pct = ((b_domain_loss - d_domain_loss) / (b_domain_loss + EPS)) * 100

    # Quality delta heatmap: averaged over layers [E, G]
    # Positive = DRO improved (lower loss)
    quality_delta = (b_q - d_q).mean(dim=0).numpy()  # [E, G]

    # Hard domain identification (by baseline loss, descending)
    domain_ranking = np.argsort(-b_domain_loss)

    # Selected-expert quality on hard domains
    # "Selected" = above-median routing mass for that domain
    L, E, G = b_m.shape
    selected_quality_baseline = {}
    selected_quality_dro = {}
    for gi, gname in enumerate(group_names):
        # Baseline selected experts
        b_mass_domain = b_m[:, :, gi].sum(dim=0)  # [E]
        median_mass = b_mass_domain.median()
        selected = b_mass_domain > median_mass
        if selected.sum() > 0:
            b_sel_loss = (b_q[:, selected, gi] * b_m[:, selected, gi]).sum() / (b_m[:, selected, gi].sum() + EPS)
            selected_quality_baseline[gname] = b_sel_loss.item()
        # DRO selected experts
        d_mass_domain = d_m[:, :, gi].sum(dim=0)
        median_mass_d = d_mass_domain.median()
        selected_d = d_mass_domain > median_mass_d
        if selected_d.sum() > 0:
            d_sel_loss = (d_q[:, selected_d, gi] * d_m[:, selected_d, gi]).sum() / (d_m[:, selected_d, gi].sum() + EPS)
            selected_quality_dro[gname] = d_sel_loss.item()

    results = {
        'domain_names': group_names,
        'baseline_domain_loss': {g: float(b_domain_loss[i]) for i, g in enumerate(group_names)},
        'dro_domain_loss': {g: float(d_domain_loss[i]) for i, g in enumerate(group_names)},
        'improvement_pct': {g: float(improvement_pct[i]) for i, g in enumerate(group_names)},
        'hard_domain_ranking': [group_names[i] for i in domain_ranking],
        'selected_expert_quality_baseline': selected_quality_baseline,
        'selected_expert_quality_dro': selected_quality_dro,
        'quality_delta': quality_delta,  # numpy, not JSON serializable
    }
    return results


def plot_experiment2(results, group_labels_map, output_dir):
    """Generate experiment 2 plots."""
    domain_names = results['domain_names']
    G = len(domain_names)
    labels = [group_labels_map.get(g, g) for g in domain_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 2a: Per-domain loss comparison
    ax = axes[0]
    x = np.arange(G)
    width = 0.35
    b_vals = [results['baseline_domain_loss'][g] for g in domain_names]
    d_vals = [results['dro_domain_loss'][g] for g in domain_names]
    imp_vals = [results['improvement_pct'][g] for g in domain_names]

    bars1 = ax.bar(x - width/2, b_vals, width, label='Baseline', color=COLOR_BASELINE)
    bars2 = ax.bar(x + width/2, d_vals, width, label='DRMoET', color=COLOR_DRO)

    for i, (b, d, imp) in enumerate(zip(b_vals, d_vals, imp_vals)):
        if imp > 0:
            ax.annotate(f'{imp:+.1f}%', xy=(i + width/2, d), xytext=(0, 5),
                       textcoords='offset points', ha='center', fontsize=8,
                       color='green', fontweight='bold')
        else:
            ax.annotate(f'{imp:+.1f}%', xy=(i + width/2, d), xytext=(0, 5),
                       textcoords='offset points', ha='center', fontsize=8,
                       color='red')

    ax.set_xlabel('Domain')
    ax.set_ylabel('Weighted Avg Expert Loss')
    ax.set_title('Per-Domain Expert Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2b: Quality delta heatmap
    ax = axes[1]
    delta = results['quality_delta']
    vmax = max(abs(delta.min()), abs(delta.max()))
    im = ax.imshow(delta, aspect='auto', cmap='RdBu', interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
    ax.set_title('Quality Improvement\n(blue = DRMoET better)')
    ax.set_xlabel('Domain')
    ax.set_ylabel('Expert ID')
    ax.set_xticks(range(G))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    plt.colorbar(im, ax=ax, label='Baseline Loss − DRMoET Loss')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(output_dir, f'experiment2_domain_quality.{fmt}'),
                    dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 3: Mid-Tier Expert Analysis
# ---------------------------------------------------------------------------

def experiment3(baseline_data, dro_data):
    """Mid-tier expert analysis."""
    b_q = baseline_data['quality']
    b_m = baseline_data['routing_mass']
    d_q = dro_data['quality']
    d_m = dro_data['routing_mass']

    L, E, G = b_m.shape
    tier_size = E // 3
    tier_names = ['Top', 'Mid', 'Bottom']

    # Per-layer tier assignment based on baseline traffic
    tier_quality_baseline = {t: [] for t in tier_names}
    tier_quality_dro = {t: [] for t in tier_names}
    tier_all_values_baseline = {t: [] for t in tier_names}
    tier_all_values_dro = {t: [] for t in tier_names}

    for l in range(L):
        # Rank experts by total traffic in baseline
        load = b_m[l].sum(dim=1)  # [E]
        ranking = torch.argsort(load, descending=True)

        tiers = {
            'Top': ranking[:tier_size],
            'Mid': ranking[tier_size:2*tier_size],
            'Bottom': ranking[2*tier_size:],
        }

        for tname, indices in tiers.items():
            # Baseline weighted quality for this tier
            b_tier_loss = (b_q[l, indices, :] * b_m[l, indices, :]).sum()
            b_tier_mass = b_m[l, indices, :].sum()
            b_wq = (b_tier_loss / (b_tier_mass + EPS)).item()
            tier_quality_baseline[tname].append(b_wq)

            # DRO weighted quality for this tier (same expert indices)
            d_tier_loss = (d_q[l, indices, :] * d_m[l, indices, :]).sum()
            d_tier_mass = d_m[l, indices, :].sum()
            d_wq = (d_tier_loss / (d_tier_mass + EPS)).item()
            tier_quality_dro[tname].append(d_wq)

            # Collect all per-expert-per-domain quality values for distributions
            for idx in indices:
                for g in range(G):
                    if b_m[l, idx, g] > EPS:
                        tier_all_values_baseline[tname].append(b_q[l, idx, g].item())
                    if d_m[l, idx, g] > EPS:
                        tier_all_values_dro[tname].append(d_q[l, idx, g].item())

    # Average across layers
    results = {
        'tier_names': tier_names,
        'tier_quality_baseline': {t: float(np.mean(v)) for t, v in tier_quality_baseline.items()},
        'tier_quality_dro': {t: float(np.mean(v)) for t, v in tier_quality_dro.items()},
        'tier_quality_per_layer_baseline': {t: v for t, v in tier_quality_baseline.items()},
        'tier_quality_per_layer_dro': {t: v for t, v in tier_quality_dro.items()},
        'tier_all_values_baseline': tier_all_values_baseline,
        'tier_all_values_dro': tier_all_values_dro,
    }

    # Improvement
    results['tier_improvement'] = {}
    results['tier_improvement_pct'] = {}
    for t in tier_names:
        b = results['tier_quality_baseline'][t]
        d = results['tier_quality_dro'][t]
        results['tier_improvement'][t] = b - d
        results['tier_improvement_pct'][t] = ((b - d) / (b + EPS)) * 100

    return results


def plot_experiment3(results, output_dir):
    """Generate experiment 3 plots."""
    tier_names = results['tier_names']
    n_tiers = len(tier_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 3a: Per-tier average quality
    ax = axes[0]
    x = np.arange(n_tiers)
    width = 0.35
    b_vals = [results['tier_quality_baseline'][t] for t in tier_names]
    d_vals = [results['tier_quality_dro'][t] for t in tier_names]
    imp_pct = [results['tier_improvement_pct'][t] for t in tier_names]

    ax.bar(x - width/2, b_vals, width, label='Baseline', color=COLOR_BASELINE)
    ax.bar(x + width/2, d_vals, width, label='DRMoET', color=COLOR_DRO)

    for i, imp in enumerate(imp_pct):
        color = 'green' if imp > 0 else 'red'
        ax.annotate(f'{imp:+.1f}%', xy=(i + width/2, d_vals[i]), xytext=(0, 5),
                   textcoords='offset points', ha='center', fontsize=9,
                   color=color, fontweight='bold')

    ax.set_xlabel('Expert Tier (by Traffic)')
    ax.set_ylabel('Weighted Avg Loss (lower = better)')
    ax.set_title('Expert Quality by Traffic Tier')
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3b: Box plot of quality distributions per tier
    ax = axes[1]
    positions = []
    data = []
    colors = []
    tick_positions = []
    tick_labels = []

    for i, t in enumerate(tier_names):
        pos_b = i * 3
        pos_d = i * 3 + 1
        b_vals_dist = results['tier_all_values_baseline'][t]
        d_vals_dist = results['tier_all_values_dro'][t]

        if b_vals_dist:
            bp = ax.boxplot([b_vals_dist], positions=[pos_b], widths=0.6,
                           patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(COLOR_BASELINE)
            bp['boxes'][0].set_alpha(0.7)
        if d_vals_dist:
            bp = ax.boxplot([d_vals_dist], positions=[pos_d], widths=0.6,
                           patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(COLOR_DRO)
            bp['boxes'][0].set_alpha(0.7)

        tick_positions.append(i * 3 + 0.5)
        tick_labels.append(t)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Per-Expert Loss')
    ax.set_title('Expert Quality Distribution by Tier')
    ax.grid(True, alpha=0.3, axis='y')

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLOR_BASELINE, alpha=0.7, label='Baseline'),
        Patch(facecolor=COLOR_DRO, alpha=0.7, label='DRMoET'),
    ])

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(output_dir, f'experiment3_midtier_analysis.{fmt}'),
                    dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Expert quality analysis: DRO vs Baseline comparison')
    parser.add_argument('--baseline', required=True,
                        help='Path to baseline domain-probe results directory')
    parser.add_argument('--dro', required=True,
                        help='Path to DRO domain-probe results directory')
    parser.add_argument('--iter', type=int, default=None,
                        help='Checkpoint iteration to analyze (default: latest)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for plots and summary')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover iterations
    b_iters = discover_iterations(args.baseline)
    d_iters = discover_iterations(args.dro)
    common_iters = sorted(set(b_iters) & set(d_iters))

    if not common_iters:
        print(f"Error: No common iterations. Baseline: {b_iters}, DRO: {d_iters}")
        return

    iteration = args.iter if args.iter else common_iters[-1]
    if iteration not in common_iters:
        print(f"Error: Iteration {iteration} not available. Common: {common_iters}")
        return

    print(f"Analyzing iteration {iteration}")
    print(f"  Baseline: {args.baseline}")
    print(f"  DRO: {args.dro}")

    baseline_data = load_raw_metrics(args.baseline, iteration)
    dro_data = load_raw_metrics(args.dro, iteration)
    layer_indices = baseline_data.get('layer_indices', list(range(baseline_data['routing_mass'].shape[0])))

    L, E, G = baseline_data['routing_mass'].shape
    print(f"  Layers: {L}, Experts: {E}, Domains: {G}")

    # Run experiments
    print("\n[Experiment 1] Router Quality...")
    exp1 = experiment1(baseline_data, dro_data)
    plot_experiment1(exp1, layer_indices, args.output_dir)
    print(f"  Baseline mean entropy: {exp1['baseline_mean_entropy']:.4f}")
    print(f"  DRO mean entropy:      {exp1['dro_mean_entropy']:.4f}")
    print(f"  Baseline mean CV:      {exp1['baseline_mean_cv']:.4f}")
    print(f"  DRO mean CV:           {exp1['dro_mean_cv']:.4f}")

    print("\n[Experiment 2] Per-Expert Domain Quality...")
    exp2 = experiment2(baseline_data, dro_data)
    plot_experiment2(exp2, DOMAIN_GROUPS, args.output_dir)
    print(f"  Per-domain improvement:")
    for g in exp2['domain_names']:
        b = exp2['baseline_domain_loss'][g]
        d = exp2['dro_domain_loss'][g]
        imp = exp2['improvement_pct'][g]
        print(f"    {g:<18s}: {b:.4f} -> {d:.4f} ({imp:+.1f}%)")
    print(f"  Hard domain ranking: {exp2['hard_domain_ranking']}")

    print("\n[Experiment 3] Mid-Tier Expert Analysis...")
    exp3 = experiment3(baseline_data, dro_data)
    plot_experiment3(exp3, args.output_dir)
    print(f"  Per-tier quality (lower = better):")
    for t in exp3['tier_names']:
        b = exp3['tier_quality_baseline'][t]
        d = exp3['tier_quality_dro'][t]
        imp = exp3['tier_improvement_pct'][t]
        print(f"    {t:<8s}: {b:.4f} -> {d:.4f} ({imp:+.1f}%)")

    # Save summary (exclude non-serializable numpy arrays)
    summary = {
        'experiment1': exp1,
        'experiment2': {k: v for k, v in exp2.items() if k != 'quality_delta'},
        'experiment3': {k: v for k, v in exp3.items()
                       if k not in ('tier_all_values_baseline', 'tier_all_values_dro',
                                    'tier_quality_per_layer_baseline', 'tier_quality_per_layer_dro')},
        'metadata': {
            'iteration': iteration,
            'n_layers': L,
            'n_experts': E,
            'n_domains': G,
            'domain_names': baseline_data.get('group_names', []),
            'baseline_path': args.baseline,
            'dro_path': args.dro,
        }
    }

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
