#!/usr/bin/env python3
"""Misrouted expert robustness probe for MoE models.

Tests whether DRO makes experts more robust by forcing tokens to the "wrong"
experts (middle-k routing) and measuring loss degradation vs normal routing.

Usage (via misroute_probe.slurm):
    MODEL_SIZE_ID=290m-seed3407-32e ITER=16000 sbatch scripts/analysis/misroute_probe.slurm
"""

import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Domain groups (same as domain_probe.py for consistency)
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

DATASET_CONFIGS = {
    'arc_easy': ('allenai/ai2_arc', 'ARC-Easy', 'validation',
                 lambda ex: f"Question: {ex['question']}\nAnswer: {ex['choices']['text'][ex['choices']['label'].index(ex['answerKey'])]}"),
    'arc_challenge': ('allenai/ai2_arc', 'ARC-Challenge', 'validation',
                      lambda ex: f"Question: {ex['question']}\nAnswer: {ex['choices']['text'][ex['choices']['label'].index(ex['answerKey'])]}"),
    'sciq': ('allenai/sciq', None, 'validation',
             lambda ex: f"Question: {ex['question']}\nAnswer: {ex['correct_answer']}"),
    'piqa': ('piqa', None, 'validation',
             lambda ex: f"Goal: {ex['goal']}\nSolution: {ex['sol1'] if ex['label'] == 0 else ex['sol2']}"),
    'hellaswag': ('hellaswag', None, 'validation',
                  lambda ex: f"{ex['ctx']} {ex['endings'][int(ex['label'])]}"),
    'record': ('super_glue', 'record', 'validation',
               lambda ex: f"{ex['passage'].replace('@highlight', '-')}\n{ex['query'].replace('@placeholder', ex['answers'][0])}"),
    'winogrande': ('allenai/winogrande', 'winogrande_xl', 'validation',
                   lambda ex: ex['sentence'].replace('_', ex['option1'] if ex['answer'] == '1' else ex['option2'])),
    'truthfulqa_mc2': ('truthful_qa', 'multiple_choice', 'validation',
                       lambda ex: f"Question: {ex['question']}\nAnswer: {ex['mc2_targets']['choices'][ex['mc2_targets']['labels'].index(1)]}"),
}


# ---------------------------------------------------------------------------
# Misrouting hooks
# ---------------------------------------------------------------------------

def install_misroute_hooks(model):
    """Install hooks that replace top-k routing with middle-k experts."""
    hooks = []
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

        def make_misroute_hook(router):
            def hook_fn(module, args, output):
                scores, routing_map = output
                with torch.no_grad():
                    input_tensor = args[0]
                    input_tensor = module.apply_input_jitter(input_tensor)
                    logits = module.gating(input_tensor)
                    seq_length, bsz = logits.shape[:2]
                    logits_flat = logits.view(-1, module.config.num_moe_experts)

                    k = module.topk
                    num_experts = logits_flat.shape[1]
                    # Sort experts by logit value, pick the middle k
                    sorted_indices = torch.argsort(logits_flat, dim=1)
                    mid_start = (num_experts - k) // 2
                    middle_indices = sorted_indices[:, mid_start:mid_start + k]

                    middle_map = torch.zeros_like(
                        logits_flat, dtype=torch.bool)
                    middle_map.scatter_(1, middle_indices, True)

                    middle_logits = torch.gather(
                        logits_flat, 1, middle_indices)
                    middle_probs = torch.softmax(
                        middle_logits.float(), dim=-1).type_as(scores)
                    middle_scores = torch.zeros_like(logits_flat).scatter(
                        1, middle_indices, middle_probs)

                return middle_scores, middle_map
            return hook_fn

        h = router.register_forward_hook(make_misroute_hook(router))
        hooks.append(h)
        print(f"    Misroute hook: layer {layer_idx}")

    print(f"  Total misroute hooks: {len(hooks)}")
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_megatron_model():
    """Load model directly via Megatron (no lm-eval-harness dependency)."""
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


def run_inference(model, tokenizer, args, texts, group_labels):
    """Run inference on texts, return per-group average loss."""
    from megatron.inference.text_generation.api import generate_and_post_process

    group_losses = defaultdict(list)

    for i, (text, group_idx) in enumerate(zip(texts, group_labels)):
        tokens = tokenizer.tokenize(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:args.seq_length]

        batch_output, batch_tokens, batch_logprobs, batch_token_ids, _ = \
            generate_and_post_process(
                model=model,
                prompts=[text],
                tokens_to_generate=0,
                return_output_log_probs=True,
                return_topk_logprobs=1,
            )

        token_logprobs = batch_logprobs[0]
        if len(token_logprobs) > 0:
            avg_loss = -sum(token_logprobs) / len(token_logprobs)
            group_losses[group_idx].append(avg_loss)

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(texts)} examples processed")

    return group_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_probe(output_dir, examples_per_group=150, seed=42, use_training_data=False,
              num_sequences=10000):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    # Step 1: Load model
    print("[1/5] Loading model...")
    model, tokenizer, args = load_megatron_model()

    # Step 2: Load probe data
    print("[2/5] Loading probe data...")
    texts = []
    group_labels = []  # 0 for all if training data (no domain groups)

    if use_training_data:
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        import glob as globmod

        data_dir = os.environ.get(
            'SSD_DATASET',
            'dataset/tokenized/EleutherAI/pythia-12b'
        )
        idx_files = sorted(globmod.glob(os.path.join(data_dir, '*.idx')))
        shard_prefixes = [f.replace('.idx', '') for f in idx_files]

        ds = IndexedDataset(shard_prefixes[0], multimodal=False, mmap=True)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        indices = indices[:num_sequences]
        print(f"  Loading {len(indices)} sequences from training data shard")

        for idx in indices:
            tokens = ds[idx]
            if len(tokens) < 2:
                continue
            tokens = tokens[:args.seq_length].tolist()
            text = tokenizer.detokenize(tokens)
            if text and len(text.strip()) > 10:
                texts.append(text)
                group_labels.append(0)
    else:
        from datasets import load_dataset
        group_to_idx = {g: i for i, g in enumerate(GROUP_NAMES)}

        for task_name, (ds_path, ds_name, split, formatter) in DATASET_CONFIGS.items():
            group_name = TASK_TO_GROUP[task_name]
            group_idx = group_to_idx[group_name]
            n_per_task = examples_per_group // len(DOMAIN_GROUPS[group_name]['tasks'])

            ds = load_dataset(ds_path, ds_name, split=split, trust_remote_code=True)
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            indices = indices[:n_per_task]

            for idx in indices:
                text = formatter(ds[int(idx)])
                if text and len(text.strip()) > 10:
                    texts.append(text)
                    group_labels.append(group_idx)

    print(f"  {len(texts)} total examples")

    # Step 3: Normal inference
    print("[3/5] Running normal inference (top-k routing)...")
    normal_losses = run_inference(model, tokenizer, args, texts, group_labels)

    # Step 4: Misrouted inference
    print("[4/5] Installing misroute hooks (middle-k routing)...")
    hooks = install_misroute_hooks(model)

    print("  Running misrouted inference...")
    misrouted_losses = run_inference(model, tokenizer, args, texts, group_labels)
    remove_hooks(hooks)

    # Step 5: Compute and report
    print("[5/5] Computing metrics...")
    print()
    print("=" * 70)
    print("MISROUTE PROBE RESULTS")
    print("=" * 70)

    header = f"{'Group':<18} {'Normal':>10} {'Misrouted':>10} {'Degrad':>10} {'Ratio':>8}"
    print(header)
    print("-" * len(header))

    all_normal = []
    all_misrouted = []
    results = {}

    if use_training_data:
        group_list = [(0, 'training_data')]
    else:
        group_list = list(enumerate(GROUP_NAMES))

    for gi, gname in group_list:
        n_losses = normal_losses.get(gi, [])
        m_losses = misrouted_losses.get(gi, [])

        if not n_losses or not m_losses:
            print(f"{gname:<18} {'N/A':>10}")
            continue

        n_avg = np.mean(n_losses)
        m_avg = np.mean(m_losses)
        degrad = m_avg - n_avg
        ratio = m_avg / n_avg if n_avg > 0 else float('inf')

        print(f"{gname:<18} {n_avg:>10.4f} {m_avg:>10.4f} {degrad:>+10.4f} {ratio:>8.3f}x")

        all_normal.extend(n_losses)
        all_misrouted.extend(m_losses)
        results[gname] = {
            'normal_loss': n_avg,
            'misrouted_loss': m_avg,
            'degradation': degrad,
            'ratio': ratio,
            'n_examples': len(n_losses),
        }

    overall_normal = np.mean(all_normal)
    overall_misrouted = np.mean(all_misrouted)
    overall_degrad = overall_misrouted - overall_normal
    overall_ratio = overall_misrouted / overall_normal

    print("-" * len(header))
    print(f"{'OVERALL':<18} {overall_normal:>10.4f} {overall_misrouted:>10.4f} {overall_degrad:>+10.4f} {overall_ratio:>8.3f}x")

    results['overall'] = {
        'normal_loss': overall_normal,
        'misrouted_loss': overall_misrouted,
        'degradation': overall_degrad,
        'ratio': overall_ratio,
        'n_examples': len(all_normal),
    }

    # Save results
    with open(os.path.join(output_dir, 'misroute_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    groups_with_data = [g for g in GROUP_NAMES if g in results]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(groups_with_data) + 1)
    width = 0.35

    normal_vals = [results[g]['normal_loss'] for g in groups_with_data] + [overall_normal]
    misrouted_vals = [results[g]['misrouted_loss'] for g in groups_with_data] + [overall_misrouted]
    labels = [DOMAIN_GROUPS[g]['label'] for g in groups_with_data] + ['OVERALL']

    bars1 = ax.bar(x - width/2, normal_vals, width, label='Normal (top-k)', color='#2196F3')
    bars2 = ax.bar(x + width/2, misrouted_vals, width, label='Misrouted (middle-k)', color='#FF5722')

    ax.set_ylabel('Avg Per-Token Loss')
    ax.set_title('Expert Robustness: Normal vs Misrouted Routing')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'misroute_comparison.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to {output_dir}")
    print("Done.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--examples-per-group', type=int, default=150)
    parser.add_argument('--probe-seed', type=int, default=42)
    parser.add_argument('--use-training-data', action='store_true')
    parser.add_argument('--num-sequences', type=int, default=10000)
    extra_args, _ = parser.parse_known_args()

    run_probe(
        output_dir=extra_args.output_dir,
        examples_per_group=extra_args.examples_per_group,
        seed=extra_args.probe_seed,
        use_training_data=extra_args.use_training_data,
        num_sequences=extra_args.num_sequences,
    )
