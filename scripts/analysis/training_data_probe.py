#!/usr/bin/env python3
"""Per-expert routing probability and loss probe on training data.

Evaluates on in-distribution (training) data to measure:
- Per-expert routing probability (does the min-probability expert get more traffic under DRO?)
- Per-expert loss (does the worst expert's loss improve under DRO?)

Usage (via training_data_probe.slurm):
    MODEL_SIZE_ID=290m-seed3407-32e ITER=16000 sbatch scripts/analysis/training_data_probe.slurm
"""

import json
import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Routing hooks (capture per-token expert selection and probabilities)
# ---------------------------------------------------------------------------

class RoutingCollector:
    """Captures per-forward-pass routing decisions from MoE routers."""

    def __init__(self, model):
        self.current_pass = {}
        self.hooks = []
        self._install(model)

    def _install(self, model):
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
                    scores, routing_map = output
                    with torch.no_grad():
                        k = module.topk
                        probs, topi = scores.float().topk(k, dim=-1)
                        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
                        self.current_pass[layer_idx] = (
                            topi.detach().cpu(),
                            probs.detach().cpu(),
                        )
                return hook_fn

            h = router.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)
            print(f"    Hook: layer {layer_idx}")
        print(f"  Total hooks: {len(self.hooks)}")

    def snapshot(self):
        snap = dict(self.current_pass)
        self.current_pass.clear()
        return snap

    def remove(self):
        for h in self.hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(output_dir, num_sequences=1000, seed=42):
    from lm_eval.models.megatron_lm import MegatronLM
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from megatron.inference.text_generation.api import generate_and_post_process

    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    # Step 1: Load model
    print("[1/5] Loading model...")
    lm = MegatronLM(batch_size=1)
    model = lm.model
    tokenizer = lm.tokenizer
    args = lm.args
    num_experts = args.num_experts
    topk = args.moe_router_topk

    # Step 2: Install routing hooks
    print("[2/5] Installing routing hooks...")
    collector = RoutingCollector(model)

    # Step 3: Load training data
    print("[3/5] Loading training data...")
    data_dir = os.environ.get(
        'SSD_DATASET',
        'dataset/tokenized/EleutherAI/pythia-12b'
    )

    # Find all shard prefixes
    import glob
    idx_files = sorted(glob.glob(os.path.join(data_dir, '*.idx')))
    shard_prefixes = [f.replace('.idx', '') for f in idx_files]

    # Load first shard and sample sequences
    ds = IndexedDataset(shard_prefixes[0], multimodal=False, mmap=True)
    total_seqs = len(ds)
    indices = list(range(total_seqs))
    rng.shuffle(indices)
    indices = indices[:num_sequences]

    print(f"  Shard: {os.path.basename(shard_prefixes[0])}")
    print(f"  Total sequences in shard: {total_seqs}")
    print(f"  Sampling {len(indices)} sequences")

    # Step 4: Run inference and collect routing + losses
    print("[4/5] Running inference on training data...")

    layer_indices = None
    # Accumulators: per-expert routing mass and loss
    routing_mass = None   # [L, E] total routing probability per expert
    loss_mass = None      # [L, E] total loss-weighted routing per expert
    token_count = None    # [L, E] number of tokens routed to each expert
    total_tokens = 0
    total_loss = 0.0

    for seq_idx, data_idx in enumerate(indices):
        tokens = ds[data_idx]
        if len(tokens) < 2:
            continue
        tokens = tokens[:args.seq_length].tolist()
        text = tokenizer.detokenize(tokens)

        collector.current_pass.clear()

        batch_output, batch_tokens, batch_logprobs, batch_token_ids, _ = \
            generate_and_post_process(
                model=model,
                inference_context=lm.inference_context,
                prompts=[text],
                tokens_to_generate=0,
                return_output_log_probs=True,
                return_topk_logprobs=1,
            )

        token_logprobs = batch_logprobs[0]
        if len(token_logprobs) == 0:
            continue
        token_losses = [-lp for lp in token_logprobs]
        num_loss_tokens = len(token_losses)
        total_tokens += num_loss_tokens
        total_loss += sum(token_losses)

        snap = collector.snapshot()
        if not snap:
            continue

        # Initialize on first pass
        if routing_mass is None:
            layer_indices = sorted(snap.keys())
            L = len(layer_indices)
            E = num_experts
            routing_mass = torch.zeros(L, E)
            loss_mass = torch.zeros(L, E)
            token_count = torch.zeros(L, E)

        for li, lidx in enumerate(layer_indices):
            if lidx not in snap:
                continue
            topi, probs = snap[lidx]  # [S, k]
            T = min(topi.shape[0] - 1, num_loss_tokens)
            topi_t = topi[:T]
            probs_t = probs[:T]
            losses_t = torch.tensor(token_losses[:T], dtype=torch.float)

            for ki in range(topi_t.shape[1]):
                eids = topi_t[:, ki].long()
                eprobs = probs_t[:, ki]
                routing_mass[li].scatter_add_(0, eids, eprobs)
                loss_mass[li].scatter_add_(0, eids, losses_t * eprobs)
                ones = torch.ones_like(eprobs)
                token_count[li].scatter_add_(0, eids, ones)

        if (seq_idx + 1) % 100 == 0:
            print(f"    {seq_idx + 1}/{len(indices)} sequences processed")

    if routing_mass is None:
        print("ERROR: No data processed!")
        return

    # Step 5: Compute and report metrics
    print("[5/5] Computing metrics...")

    L, E = routing_mass.shape
    # Normalize routing mass to get per-expert routing fraction
    routing_frac = routing_mass / (routing_mass.sum(dim=1, keepdim=True) + 1e-8)  # [L, E]
    # Per-expert average loss
    expert_avg_loss = loss_mass / (routing_mass + 1e-8)  # [L, E]

    # Average across layers
    avg_frac = routing_frac.mean(dim=0)  # [E]
    avg_loss = expert_avg_loss.mean(dim=0)  # [E]
    avg_count = token_count.mean(dim=0)  # [E]

    # Key metrics
    min_frac_idx = avg_frac.argmin().item()
    max_frac_idx = avg_frac.argmax().item()
    min_frac = avg_frac[min_frac_idx].item()
    max_frac = avg_frac[max_frac_idx].item()
    uniform_frac = 1.0 / E

    worst_loss_idx = avg_loss.argmax().item()
    best_loss_idx = avg_loss.argmin().item()
    worst_loss = avg_loss[worst_loss_idx].item()
    best_loss = avg_loss[best_loss_idx].item()

    # Routing entropy
    ent = -(avg_frac * torch.log(avg_frac + 1e-10)).sum().item()
    max_ent = np.log(E)

    # Gini coefficient
    sorted_frac = torch.sort(avg_frac)[0]
    n = len(sorted_frac)
    index = torch.arange(1, n + 1, dtype=torch.float)
    gini = (2 * (index * sorted_frac).sum() / (n * sorted_frac.sum()) - (n + 1) / n).item()

    avg_total_loss = total_loss / total_tokens if total_tokens > 0 else 0

    print()
    print("=" * 70)
    print("TRAINING DATA PROBE RESULTS")
    print("=" * 70)
    print(f"  Sequences: {len(indices)}, Tokens: {total_tokens}")
    print(f"  Avg loss: {avg_total_loss:.4f}")
    print(f"  Experts: {E}, Top-k: {topk}, MoE layers: {layer_indices}")
    print()
    print("--- Routing Probability ---")
    print(f"  Uniform fraction: {uniform_frac:.5f}")
    print(f"  Min-prob expert: E{min_frac_idx} ({min_frac:.5f}, {min_frac/uniform_frac:.2f}x uniform)")
    print(f"  Max-prob expert: E{max_frac_idx} ({max_frac:.5f}, {max_frac/uniform_frac:.2f}x uniform)")
    print(f"  Max/Min ratio: {max_frac/min_frac:.2f}")
    print(f"  Std: {avg_frac.std():.5f}")
    print(f"  Gini: {gini:.4f}")
    print(f"  Entropy: {ent:.4f} (max: {max_ent:.4f})")
    print()
    print("--- Per-Expert Loss ---")
    print(f"  Best expert:  E{best_loss_idx} (loss={best_loss:.4f})")
    print(f"  Worst expert: E{worst_loss_idx} (loss={worst_loss:.4f})")
    print(f"  Loss range: {worst_loss - best_loss:.4f}")
    print(f"  Loss std: {avg_loss.std():.4f}")
    print(f"  Loss CV: {avg_loss.std() / avg_loss.mean():.4f}")
    print()
    print("--- Min-Prob Expert's Loss ---")
    min_expert_loss = avg_loss[min_frac_idx].item()
    print(f"  E{min_frac_idx}: routing_frac={min_frac:.5f}, loss={min_expert_loss:.4f}")
    print()
    print("--- Per-Expert Detail (sorted by routing fraction) ---")
    order = avg_frac.argsort()
    print(f"  {'Expert':>8} {'Frac':>10} {'Loss':>10} {'Tokens/L':>10}")
    for e in order:
        e = e.item()
        print(f"  E{e:2d}      {avg_frac[e]:>10.5f} {avg_loss[e]:>10.4f} {avg_count[e]:>10.1f}")

    # Per-layer min expert
    print()
    print("--- Per-Layer Min-Prob Expert ---")
    for li, lidx in enumerate(layer_indices):
        frac_l = routing_frac[li]
        loss_l = expert_avg_loss[li]
        min_e = frac_l.argmin().item()
        print(f"  Layer {lidx}: E{min_e:2d}, frac={frac_l[min_e]:.5f}, loss={loss_l[min_e]:.4f}")

    # Save results
    results = {
        'num_sequences': len(indices),
        'total_tokens': total_tokens,
        'avg_loss': avg_total_loss,
        'num_experts': E,
        'topk': topk,
        'moe_layers': layer_indices,
        'uniform_fraction': uniform_frac,
        'min_frac_expert': min_frac_idx,
        'min_frac': min_frac,
        'max_frac_expert': max_frac_idx,
        'max_frac': max_frac,
        'max_min_ratio': max_frac / min_frac,
        'routing_std': avg_frac.std().item(),
        'routing_gini': gini,
        'routing_entropy': ent,
        'max_entropy': max_ent,
        'best_loss_expert': best_loss_idx,
        'best_loss': best_loss,
        'worst_loss_expert': worst_loss_idx,
        'worst_loss': worst_loss,
        'loss_range': worst_loss - best_loss,
        'loss_std': avg_loss.std().item(),
        'loss_cv': (avg_loss.std() / avg_loss.mean()).item(),
        'min_expert_loss': min_expert_loss,
        'per_expert_frac': avg_frac.tolist(),
        'per_expert_loss': avg_loss.tolist(),
    }

    with open(os.path.join(output_dir, 'training_probe_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    torch.save({
        'routing_mass': routing_mass,
        'loss_mass': loss_mass,
        'token_count': token_count,
        'routing_frac': routing_frac,
        'expert_avg_loss': expert_avg_loss,
        'layer_indices': layer_indices,
    }, os.path.join(output_dir, 'training_probe_raw.pt'))

    # Plot: routing fraction vs expert loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: routing fraction per expert (bar chart, sorted)
    order_np = avg_frac.argsort().numpy()
    ax = axes[0]
    ax.bar(range(E), avg_frac[order_np].numpy(), color='#2196F3')
    ax.axhline(y=uniform_frac, color='red', linestyle='--', label=f'Uniform (1/{E})')
    ax.set_xlabel('Expert (sorted by routing fraction)')
    ax.set_ylabel('Routing Fraction')
    ax.set_title('Per-Expert Routing Probability')
    ax.legend()

    # Right: expert loss vs routing fraction (scatter)
    ax = axes[1]
    ax.scatter(avg_frac.numpy(), avg_loss.numpy(), alpha=0.7, c='#FF5722')
    ax.set_xlabel('Routing Fraction')
    ax.set_ylabel('Avg Expert Loss')
    ax.set_title('Expert Loss vs Routing Fraction')
    # Annotate min and max
    ax.annotate(f'E{min_frac_idx}', (min_frac, min_expert_loss),
                fontsize=9, ha='right')
    ax.annotate(f'E{max_frac_idx}', (max_frac, avg_loss[max_frac_idx].item()),
                fontsize=9, ha='left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_probe.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to {output_dir}")
    print("Done.")

    collector.remove()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--num-sequences', type=int, default=1000)
    parser.add_argument('--probe-seed', type=int, default=42)
    extra_args, _ = parser.parse_known_args()

    run_probe(
        output_dir=extra_args.output_dir,
        num_sequences=extra_args.num_sequences,
        seed=extra_args.probe_seed,
    )
