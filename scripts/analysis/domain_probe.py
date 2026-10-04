#!/usr/bin/env python3
"""Domain-labeled expert specialization probe for MoE models.

Uses lm-evaluation-harness for correct teacher-forced inference, with routing
hooks installed on MoE routers to capture per-token expert selection.

Usage (via domain_probe.slurm):
    MODEL_SIZE_ID=1.7b_seed7 ITER=11029 sbatch scripts/analysis/domain_probe.slurm
"""

import json
import os
import sys
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Domain group definitions
# ---------------------------------------------------------------------------
DOMAIN_GROUPS = {
    'science': {
        'tasks': ['arc_easy', 'arc_challenge', 'sciq'],
        'label': 'Science / Factual',
    },
    'physical': {
        'tasks': ['piqa'],
        'label': 'Physical Commonsense',
    },
    'narrative': {
        'tasks': ['hellaswag'],
        'label': 'Narrative / Event',
    },
    'reading': {
        'tasks': ['record'],
        'label': 'Reading Comprehension',
    },
    'coreference': {
        'tasks': ['winogrande'],
        'label': 'Coreference / Commonsense',
    },
    'truthfulness': {
        'tasks': ['truthfulqa_mc2'],
        'label': 'Truthfulness',
    },
}

GROUP_NAMES = list(DOMAIN_GROUPS.keys())
TASK_TO_GROUP = {}
for gname, ginfo in DOMAIN_GROUPS.items():
    for task in ginfo['tasks']:
        TASK_TO_GROUP[task] = gname


# ---------------------------------------------------------------------------
# Routing hooks
# ---------------------------------------------------------------------------

class RoutingCollector:
    """Installs hooks on MoE routers and collects per-forward-pass routing data."""

    def __init__(self, model):
        self.routing_data = {}  # layer_idx -> list of (topi, probs) per forward
        self.current_pass = {}  # layer_idx -> (topi, probs) for current forward
        self.hooks = []
        self._install_hooks(model)

    def _install_hooks(self, model):
        for name, module in model.named_modules():
            if not hasattr(module, 'router'):
                continue
            router = module.router
            parts = name.split('.')
            layer_idx = None
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None:
                continue

            def make_hook(layer_idx):
                def hook_fn(module, args, output):
                    # output: (scores [S*B, E], routing_map [S*B, E])
                    scores, routing_map = output
                    with torch.no_grad():
                        k = module.topk
                        probs, topi = scores.float().topk(k, dim=-1)  # [S*B, k]
                        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
                        self.current_pass[layer_idx] = (
                            topi.detach().cpu(),
                            probs.detach().cpu(),
                        )
                return hook_fn

            h = router.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)
            print(f"    Hook: layer {layer_idx} ({name}.router)")

        print(f"  Total hooks: {len(self.hooks)}")

    def snapshot(self):
        """Take a snapshot of current routing pass data and return it."""
        snap = {}
        for layer_idx, (topi, probs) in self.current_pass.items():
            snap[layer_idx] = (topi, probs)
        self.current_pass.clear()
        return snap

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Run evaluation with lm-eval-harness and capture routing
# ---------------------------------------------------------------------------

def load_megatron_model():
    """Load model directly via Megatron."""
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args, get_tokenizer, get_model
    from megatron.training.checkpointing import load_checkpoint
    from pretrain_gpt import model_provider

    initialize_megatron(
        args_defaults={'no_load_rng': True, 'no_load_optim': True,
                       'exit_on_missing_checkpoint': True},
        ignore_unknown_args=True,
    )
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]
    model.eval()
    return model, get_tokenizer(), get_args()


def run_probe(output_dir, examples_per_group=150, seed=42):
    """Run the domain probe."""

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Initialize model
    print("[1/5] Loading model...")
    model, tokenizer, args = load_megatron_model()

    # Step 2: Install routing hooks
    print("[2/5] Installing routing hooks...")
    collector = RoutingCollector(model)

    # Step 3: Run inference with routing capture
    print("[3/5] Running inference with routing capture...")
    from datasets import load_dataset

    rng = random.Random(seed)

    DATASET_CONFIGS = {
        'arc_easy': ('allenai/ai2_arc', 'ARC-Easy', ['train', 'validation'],
                     lambda ex: f"Question: {ex['question']}\nAnswer: {ex['choices']['text'][ex['choices']['label'].index(ex['answerKey'])]}"),
        'arc_challenge': ('allenai/ai2_arc', 'ARC-Challenge', ['train', 'validation'],
                          lambda ex: f"Question: {ex['question']}\nAnswer: {ex['choices']['text'][ex['choices']['label'].index(ex['answerKey'])]}"),
        'sciq': ('allenai/sciq', None, ['train', 'validation'],
                 lambda ex: f"Question: {ex['question']}\nAnswer: {ex['correct_answer']}"),
        'piqa': ('piqa', None, ['train', 'validation'],
                 lambda ex: f"Goal: {ex['goal']}\nSolution: {ex['sol1'] if ex['label'] == 0 else ex['sol2']}"),
        'hellaswag': ('hellaswag', None, ['train', 'validation'],
                      lambda ex: f"{ex['ctx']} {ex['endings'][int(ex['label'])]}"),
        'record': ('super_glue', 'record', ['train', 'validation'],
                   lambda ex: f"{ex['passage'].replace('@highlight', '-')}\n{ex['query'].replace('@placeholder', ex['answers'][0])}"),
        'winogrande': ('allenai/winogrande', 'winogrande_xl', ['train', 'validation'],
                       lambda ex: ex['sentence'].replace('_', ex['option1'] if ex['answer'] == '1' else ex['option2'])),
        'truthfulqa_mc2': ('truthful_qa', 'multiple_choice', ['validation'],
                           lambda ex: f"Question: {ex['question']}\nAnswer: {ex['mc2_targets']['choices'][ex['mc2_targets']['labels'].index(1)]}"),
    }

    # Accumulate routing per group
    L = None  # number of MoE layers (discovered from first forward)
    E = args.num_experts
    G = len(GROUP_NAMES)
    group_to_idx = {g: i for i, g in enumerate(GROUP_NAMES)}

    routing_mass = None
    expert_loss_mass = None
    group_token_counts = torch.zeros(G)

    # tokenizer already loaded above

    for task_name, (ds_path, ds_name, splits, formatter) in DATASET_CONFIGS.items():
        group_name = TASK_TO_GROUP[task_name]
        group_idx = group_to_idx[group_name]

        # Load and concatenate all splits
        from datasets import concatenate_datasets
        ds_parts = []
        for split in splits:
            ds_parts.append(load_dataset(ds_path, ds_name, split=split, trust_remote_code=True))
        ds = concatenate_datasets(ds_parts)

        indices = list(range(len(ds)))
        rng.shuffle(indices)
        n_per_task = examples_per_group // len(DOMAIN_GROUPS[group_name]['tasks'])
        indices = indices[:n_per_task]

        print(f"    {task_name} ({group_name}): {len(indices)} examples (of {len(ds)} available)")

        for ex_i, idx in enumerate(indices):
            if (ex_i + 1) % 500 == 0:
                print(f"      {ex_i + 1}/{len(indices)} processed")
            example = ds[int(idx)]
            text = formatter(example)
            if not text or len(text.strip()) < 10:
                continue

            tokens = tokenizer.tokenize(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:args.seq_length]

            input_ids = torch.tensor([tokens], dtype=torch.long, device='cuda')
            seq_len = input_ids.shape[1]

            collector.current_pass.clear()

            # Direct model forward (faster than generate_and_post_process)
            position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0)
            attention_mask = torch.ones(1, 1, seq_len, seq_len,
                                       device='cuda', dtype=torch.bool).tril()
            with torch.no_grad():
                output = model(input_ids, position_ids, attention_mask)
                logits_out = output.float()
                shift_logits = logits_out[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                token_losses = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none',
                ).cpu()

            # Get routing snapshot
            snap = collector.snapshot()

            if not snap:
                continue

            # Initialize accumulators on first pass
            if routing_mass is None:
                layer_indices = sorted(snap.keys())
                L = len(layer_indices)
                routing_mass = torch.zeros(L, E, G)
                expert_loss_mass = torch.zeros(L, E, G)

            num_loss_tokens = len(token_losses)
            group_token_counts[group_idx] += num_loss_tokens

            for li, layer_idx in enumerate(layer_indices):
                if layer_idx not in snap:
                    continue
                topi, probs = snap[layer_idx]  # [S, k] each
                # Align with loss tokens (loss is shifted by 1)
                T = min(topi.shape[0] - 1, num_loss_tokens)
                topi_t = topi[:T]   # [T, k]
                probs_t = probs[:T]  # [T, k]
                losses_t = token_losses[:T]  # [T]

                for k_idx in range(topi_t.shape[1]):
                    expert_ids = topi_t[:, k_idx].long()  # [T]
                    expert_probs = probs_t[:, k_idx]  # [T]
                    weighted_loss = losses_t * expert_probs

                    routing_mass[li, :, group_idx].scatter_add_(
                        0, expert_ids, expert_probs)
                    expert_loss_mass[li, :, group_idx].scatter_add_(
                        0, expert_ids, weighted_loss)

    if routing_mass is None:
        print("ERROR: No routing data captured!")
        return

    # Compute metrics
    eps = 1e-8
    total_tokens = group_token_counts.sum()
    pi = group_token_counts / (total_tokens + eps)  # [G]

    # Selectivity: S_{l,i,g} = (m_{l,i,g} / sum_g' m_{l,i,g'}) / pi_g
    mass_per_expert = routing_mass.sum(dim=2, keepdim=True)  # [L, E, 1]
    frac = routing_mass / (mass_per_expert + eps)
    selectivity = frac / (pi.unsqueeze(0).unsqueeze(0) + eps)  # [L, E, G]

    # Quality: Q_{l,i,g} = expert_loss_mass / routing_mass
    quality = expert_loss_mass / (routing_mass + eps)  # [L, E, G]

    # Specialization metrics
    sorted_S, _ = selectivity.sort(dim=2, descending=True)
    delta_S = sorted_S[:, :, 0] - sorted_S[:, :, 1]
    preferred = selectivity.argmax(dim=2)
    Q_preferred = quality.gather(2, preferred.unsqueeze(2)).squeeze(2)
    Q_mean_others = (quality.sum(dim=2) - Q_preferred) / max(G - 1, 1)
    delta_Q = Q_mean_others - Q_preferred

    # Mutual information per layer
    mi_per_layer = []
    for l in range(L):
        m = routing_mass[l]  # [E, G]
        p_eg = m / (m.sum() + eps)
        p_e = p_eg.sum(dim=1, keepdim=True)
        p_g = p_eg.sum(dim=0, keepdim=True)
        log_ratio = torch.log(p_eg / (p_e * p_g + 1e-12) + 1e-12)
        mi = (p_eg * log_ratio).sum()
        mi_per_layer.append(mi.item())

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  MoE layers: {layer_indices}")
    print(f"  Experts: {E}, Groups: {GROUP_NAMES}")
    print(f"  Tokens per group: {group_token_counts.tolist()}")
    print(f"  Avg max selectivity: {selectivity.max(dim=2)[0].mean():.4f}")
    print(f"  Avg specialization margin (ΔS): {delta_S.mean():.4f}")
    print(f"  Avg competence advantage (ΔQ): {delta_Q.mean():.4f}")
    print(f"  Avg mutual information I(E;G): {np.mean(mi_per_layer):.4f}")
    print(f"  MI per layer: {[f'{v:.4f}' for v in mi_per_layer]}")

    # Save
    spec_metrics = {
        'avg_max_selectivity': selectivity.max(dim=2)[0].mean().item(),
        'avg_delta_S': delta_S.mean().item(),
        'avg_delta_Q': delta_Q.mean().item(),
        'avg_mutual_information': np.mean(mi_per_layer),
        'mi_per_layer': mi_per_layer,
        'moe_layers': layer_indices,
        'group_names': GROUP_NAMES,
        'group_labels': [DOMAIN_GROUPS[g]['label'] for g in GROUP_NAMES],
        'group_token_counts': group_token_counts.tolist(),
    }

    torch.save({
        'routing_mass': routing_mass,
        'selectivity': selectivity,
        'quality': quality,
        'layer_indices': layer_indices,
        'group_names': GROUP_NAMES,
        'group_labels': [DOMAIN_GROUPS[g]['label'] for g in GROUP_NAMES],
        'group_token_counts': group_token_counts,
        'specialization_metrics': spec_metrics,
    }, os.path.join(output_dir, 'raw_metrics.pt'))

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(spec_metrics, f, indent=2)

    # Heatmaps
    print("\nGenerating heatmaps...")
    group_labels = [DOMAIN_GROUPS[g]['label'] for g in GROUP_NAMES]

    # Average across layers
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    S_avg = selectivity.mean(dim=0).numpy()
    im0 = axes[0].imshow(S_avg, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_title('Avg Selectivity S (all MoE layers)')
    axes[0].set_xlabel('Domain Group')
    axes[0].set_ylabel('Expert ID')
    axes[0].set_xticks(range(G))
    axes[0].set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
    plt.colorbar(im0, ax=axes[0])

    Q_avg = quality.mean(dim=0).numpy()
    im1 = axes[1].imshow(Q_avg, aspect='auto', cmap='YlOrRd_r', interpolation='nearest')
    axes[1].set_title('Avg Normalized Loss Q\n(lower = more competent)')
    axes[1].set_xlabel('Domain Group')
    axes[1].set_ylabel('Expert ID')
    axes[1].set_xticks(range(G))
    axes[1].set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_avg.png'), dpi=150)
    plt.close()

    # Per-layer heatmaps
    for li, layer_idx in enumerate(layer_indices):
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        S = selectivity[li].numpy()
        im0 = axes[0].imshow(S, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        axes[0].set_title(f'Selectivity S (Layer {layer_idx})')
        axes[0].set_xlabel('Domain Group')
        axes[0].set_ylabel('Expert ID')
        axes[0].set_xticks(range(G))
        axes[0].set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
        plt.colorbar(im0, ax=axes[0])

        Q = quality[li].numpy()
        im1 = axes[1].imshow(Q, aspect='auto', cmap='YlOrRd_r', interpolation='nearest')
        axes[1].set_title(f'Normalized Loss Q (Layer {layer_idx})')
        axes[1].set_xlabel('Domain Group')
        axes[1].set_ylabel('Expert ID')
        axes[1].set_xticks(range(G))
        axes[1].set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_layer_{layer_idx}.png'), dpi=150)
        plt.close()

    print(f"\nResults saved to {output_dir}")
    print("Done.")

    collector.remove_hooks()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--examples-per-group', type=int, default=150)
    parser.add_argument('--probe-seed', type=int, default=42)
    extra_args, _ = parser.parse_known_args()

    run_probe(
        output_dir=extra_args.output_dir,
        examples_per_group=extra_args.examples_per_group,
        seed=extra_args.probe_seed,
    )
