#!/usr/bin/env python3
"""
Count parameters in a Megatron distributed checkpoint.

Usage:
    python scripts/count_parameters.py /path/to/checkpoint/iter_XXXXXX [--top-k 6]

Example:
    python scripts/count_parameters.py weights/290m/iter_0005473 --top-k 6
"""

import argparse
import pickle
from pathlib import Path
from collections import defaultdict
import re
import sys

# Add Megatron-LM to path for unpickling metadata
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "Megatron-LM"))
sys.path.insert(0, str(SCRIPT_DIR / "apex"))


def find_checkpoint_dir(ckpt_path: Path, iteration: int = None) -> Path:
    """Find the actual checkpoint directory (handles top-level dirs with iter_XXXXX subdirs)."""
    metadata_file = ckpt_path / ".metadata"
    if metadata_file.exists():
        return ckpt_path

    # Look for iter_XXXXX subdirectories
    iter_dirs = sorted(ckpt_path.glob("iter_*"))
    if iter_dirs:
        if iteration is not None:
            # Find specific iteration
            target = ckpt_path / f"iter_{iteration:07d}"
            if target.exists():
                print(f"Using checkpoint: {target.name}")
                return target
            else:
                raise FileNotFoundError(f"Iteration {iteration} not found in: {ckpt_path}")
        else:
            # Use latest
            latest = iter_dirs[-1]
            print(f"Found {len(iter_dirs)} checkpoints, using latest: {latest.name}")
            return latest

    raise FileNotFoundError(f"No checkpoint found in: {ckpt_path}")


def load_metadata(ckpt_path: Path, iteration: int = None):
    """Load checkpoint metadata without needing distributed setup."""
    ckpt_path = find_checkpoint_dir(ckpt_path, iteration)
    metadata_file = ckpt_path / ".metadata"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    return metadata


def count_parameters(ckpt_path: Path, top_k: int = 6, num_experts: int = 64, iteration: int = None):
    """
    Count total and activated parameters in a Megatron MoE checkpoint.

    Args:
        ckpt_path: Path to checkpoint directory (e.g., weights/290m/iter_0005473)
        top_k: Number of experts activated per token (for activated param count)
        num_experts: Total number of experts per MoE layer
        iteration: Specific iteration to load (default: latest)

    Returns:
        Dictionary with parameter counts
    """
    metadata = load_metadata(ckpt_path, iteration)

    total_params = 0
    expert_params_per_layer = defaultdict(int)  # layer_idx -> total expert params
    non_expert_params = 0
    shared_expert_params = 0

    # Pattern to match expert parameters: mlp.experts.experts.linear_fc...
    expert_pattern = re.compile(r'mlp\.experts\.experts\.linear_fc')
    # Pattern to match shared expert parameters
    shared_expert_pattern = re.compile(r'shared_expert\.')
    # Pattern to extract layer number
    layer_pattern = re.compile(r'decoder\.layers\.(\d+)\.')

    for key, tensor_meta in metadata.state_dict_metadata.items():
        if not hasattr(tensor_meta, 'size'):
            continue  # Skip non-tensor entries

        # Calculate number of parameters
        size = tensor_meta.size
        num_params = 1
        for dim in size:
            num_params *= dim

        total_params += num_params

        # Check if this is an expert parameter
        expert_match = expert_pattern.search(key)
        shared_match = shared_expert_pattern.search(key)
        layer_match = layer_pattern.search(key)

        # Skip _extra_state entries and optimizer states (they're not model weights)
        if '_extra_state' in key or 'optimizer.state' in key or key.startswith('chained_'):
            total_params -= num_params  # Don't count these
            continue

        if shared_match:
            shared_expert_params += num_params
        elif expert_match and layer_match:
            layer_idx = int(layer_match.group(1))
            expert_params_per_layer[layer_idx] += num_params
        else:
            non_expert_params += num_params

    # Calculate activated parameters
    # Activated = non-expert + shared_expert + (top_k / num_experts) * expert_params
    total_expert_params = sum(expert_params_per_layer.values())
    num_moe_layers = len(expert_params_per_layer)

    if num_moe_layers > 0:
        # Expert params per layer (assuming uniform distribution)
        expert_params_per_moe_layer = total_expert_params / num_moe_layers
        # Activated expert params = top_k experts per layer
        activated_expert_params = (top_k / num_experts) * total_expert_params
    else:
        activated_expert_params = 0

    activated_params = non_expert_params + shared_expert_params + activated_expert_params

    return {
        'total_params': total_params,
        'activated_params': activated_params,
        'non_expert_params': non_expert_params,
        'shared_expert_params': shared_expert_params,
        'total_expert_params': total_expert_params,
        'num_moe_layers': num_moe_layers,
        'top_k': top_k,
        'num_experts': num_experts,
    }


def format_params(n: float) -> str:
    """Format parameter count in human-readable form."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(int(n))


def main():
    parser = argparse.ArgumentParser(description="Count parameters in Megatron checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint directory (or parent with iter_XXXXX)")
    parser.add_argument("--top-k", type=int, default=6, help="Number of experts activated per token")
    parser.add_argument("--num-experts", type=int, default=64, help="Total number of experts")
    parser.add_argument("--iter", type=int, default=None, help="Specific iteration to load (default: latest)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint path does not exist: {args.checkpoint}")
        return 1

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Top-k: {args.top_k}, Num experts: {args.num_experts}")
    print("-" * 50)

    counts = count_parameters(args.checkpoint, args.top_k, args.num_experts, args.iter)

    print(f"\nParameter Counts:")
    print(f"  Total parameters:        {format_params(counts['total_params']):>10} ({counts['total_params']:,})")
    print(f"  Activated parameters:    {format_params(counts['activated_params']):>10} ({counts['activated_params']:,.0f})")
    print(f"\nBreakdown:")
    print(f"  Non-expert parameters:   {format_params(counts['non_expert_params']):>10} ({counts['non_expert_params']:,})")
    print(f"  Shared expert parameters:{format_params(counts['shared_expert_params']):>10} ({counts['shared_expert_params']:,})")
    print(f"  Total expert parameters: {format_params(counts['total_expert_params']):>10} ({counts['total_expert_params']:,})")
    print(f"  Number of MoE layers:    {counts['num_moe_layers']}")

    if counts['num_moe_layers'] > 0:
        print(f"\nMoE Statistics:")
        print(f"  Experts per layer:       {args.num_experts}")
        print(f"  Top-k routing:           {args.top_k}")
        print(f"  Expert utilization:      {args.top_k/args.num_experts*100:.1f}%")

    return 0


if __name__ == "__main__":
    exit(main())
