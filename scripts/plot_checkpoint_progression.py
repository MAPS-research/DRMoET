#!/usr/bin/env python3
"""
Plot evaluation metrics progression across checkpoints.

Usage:
    python scripts/plot_checkpoint_progression.py <csv_file>
    python scripts/plot_checkpoint_progression.py logs/evaluate-local/12345/results_summary.csv

Multiple CSV files can be provided to compare different models:
    python scripts/plot_checkpoint_progression.py file1.csv file2.csv file3.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(filepath: str) -> tuple[pd.DataFrame, str]:
    """Load CSV file and extract model name from header comment or filename."""
    filepath = Path(filepath)

    # Try to extract model name from first line comment
    model_name = None
    with open(filepath) as f:
        first_line = f.readline().strip()
        if first_line.startswith("# MODEL_SIZE_ID:"):
            model_name = first_line.split(":", 1)[1].strip()

    # Fall back to filename or parent directory
    if not model_name:
        # Use parent directory name, removing timestamp suffix
        parent_name = filepath.parent.name
        # Remove timestamp like _20260110_031926
        import re
        model_name = re.sub(r'_\d{8}_\d{6}$', '', parent_name)
        # Shorten common prefixes for readability
        model_name = model_name.replace('290m-dro-', '')

    # Load CSV, skipping comment lines
    df = pd.read_csv(filepath, comment="#")

    return df, model_name


def plot_single_model(csv_file: str, output_path: str = None, show: bool = True):
    """Plot metrics for a single model's CSV file."""

    # Define which metrics to plot and their display names
    metrics = {
        "arc_challenge": "ARC-Challenge",
        "arc_easy": "ARC-Easy",
        "hellaswag": "HellaSwag",
        "winogrande": "WinoGrande",
        "piqa": "PIQA",
        "openbookqa": "OpenBookQA",
        "sciq": "SciQ",
        "record": "ReCoRD (F1)",
        "truthfulqa_mc2": "TruthfulQA MC2",
    }

    try:
        df, model_name = load_csv(csv_file)
        print(f"Loaded: {csv_file} ({model_name}, {len(df)} checkpoints)")
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None

    # Create figure with subplots
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    fig.suptitle(model_name, fontsize=14, fontweight='bold')

    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes[idx]

        if metric_key in df.columns:
            ax.plot(df["iter"], df[metric_key],
                   marker="o", markersize=6, linewidth=2,
                   color="tab:blue")

            # Find and annotate the maximum point
            max_idx = df[metric_key].idxmax()
            max_iter = df.loc[max_idx, "iter"]
            max_val = df.loc[max_idx, metric_key]
            ax.annotate(f"{max_val:.4f}",
                       xy=(max_iter, max_val),
                       xytext=(5, 5), textcoords="offset points",
                       fontsize=9, color="tab:red", fontweight="bold")
            ax.scatter([max_iter], [max_val], color="tab:red", s=80, zorder=5)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy")
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save to the same directory as the CSV if no output path specified
    if output_path is None:
        output_path = str(Path(csv_file).parent / "checkpoint_progression.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_metrics(csv_files: list[str], output_path: str = None, show: bool = True):
    """Plot metrics for each CSV file separately, saving to each model's directory."""

    for csv_file in csv_files:
        plot_single_model(csv_file, output_path=None, show=show)


def plot_average(csv_files: list[str], output_path: str = None, show: bool = True):
    """Plot average accuracy across all metrics."""

    metrics_to_avg = [
        "arc_challenge", "arc_easy", "hellaswag",
        "winogrande", "piqa", "openbookqa"
    ]

    # Load all CSV files
    data = []
    for csv_file in csv_files:
        try:
            df, model_name = load_csv(csv_file)
            # Calculate average
            available_metrics = [m for m in metrics_to_avg if m in df.columns]
            df["avg_accuracy"] = df[available_metrics].mean(axis=1)
            data.append((df, model_name))
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (df, model_name) in enumerate(data):
        color = colors[i % len(colors)]
        ax.plot(df["iter"], df["avg_accuracy"],
               marker="o", markersize=5,
               label=model_name, color=color, linewidth=2)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title("Average Accuracy Across All Benchmarks", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if output_path:
        avg_output = output_path.replace(".png", "_average.png")
        plt.savefig(avg_output, dpi=150, bbox_inches="tight")
        print(f"Saved average plot to: {avg_output}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation metrics progression across checkpoints"
    )
    parser.add_argument(
        "csv_files", nargs="+",
        help="CSV file(s) with evaluation results"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path for the plot (e.g., plot.png)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display the plot (useful for batch processing)"
    )
    parser.add_argument(
        "--average-only", action="store_true",
        help="Only plot the average accuracy"
    )

    args = parser.parse_args()

    show = not args.no_show

    if args.average_only:
        plot_average(args.csv_files, args.output, show)
    else:
        plot_metrics(args.csv_files, args.output, show)


if __name__ == "__main__":
    main()
