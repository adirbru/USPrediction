#!/usr/bin/env python3
"""
Visualize results from leave-one-subject-out cross-validation experiment.
Creates bar graphs and line graphs for metrics across subjects.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    """Load all subject results from JSON files"""
    results_dir = Path(results_dir)
    all_results = {}

    # Find all subject result files
    for json_file in sorted(results_dir.glob("subject_*_results.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            subject_id = data['subject_id']
            all_results[subject_id] = data

    return all_results


def create_bar_graph(results, output_path):
    """Create bar graph with four columns (one for each metric) and one bar for each subject"""
    subjects = sorted(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'loss']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'Loss']

    # Extract validation metrics for each subject
    data = {metric: [] for metric in metrics}
    for subject in subjects:
        for metric in metrics:
            data[metric].append(results[subject]['val_metrics'][metric])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set up bar positions
    x = np.arange(len(subjects))
    width = 0.2

    # Create bars for each metric
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, data[metric], width, label=label)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    # Customize plot
    ax.set_xlabel('Subject ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Validation Metrics by Subject (Leave-One-Out Cross-Validation)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved bar graph to {output_path}")


def create_line_graphs(results, output_path):
    """Create four line graphs (one for each metric) where each subject is a different line"""
    subjects = sorted(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'loss']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'Loss']

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # For each metric, create a line graph
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # Extract data for this metric across all subjects
        values = [results[subject]['val_metrics'][metric] for subject in subjects]

        # Create line plot
        x = np.arange(len(subjects))
        ax.plot(x, values, marker='o', linewidth=2, markersize=8, label=label, color=f"C{idx}")

        # Add value labels on points
        for i, (xi, yi) in enumerate(zip(x, values)):
            ax.text(xi, yi, f'{yi:.3f}', ha='center', va='bottom', fontsize=9)

        # Customize subplot
        ax.set_xlabel('Subject ID', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Across Subjects', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.grid(True, alpha=0.3)

        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_val:.3f}')
        ax.legend(fontsize=9)

    plt.suptitle('Validation Metrics Across Subjects (Leave-One-Out Cross-Validation)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved line graphs to {output_path}")


def print_summary_statistics(results):
    """Print summary statistics for all metrics"""
    subjects = sorted(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'loss']

    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Validation Metrics)")
    print("="*80)

    for metric in metrics:
        values = [results[subject]['val_metrics'][metric] for subject in subjects]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {mean_val:.4f}")
        print(f"  Std:    {std_val:.4f}")
        print(f"  Min:    {min_val:.4f}")
        print(f"  Max:    {max_val:.4f}")

    print("\n" + "="*80)


def create_combined_visualization(results, output_path):
    """Create a comprehensive visualization combining bar and line graphs"""
    subjects = sorted(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'loss']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'Loss']

    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Top: Large bar graph spanning both columns
    ax_bar = fig.add_subplot(gs[0, :])

    # Extract validation metrics for each subject
    data = {metric: [] for metric in metrics}
    for subject in subjects:
        for metric in metrics:
            data[metric].append(results[subject]['val_metrics'][metric])

    # Create bars for each metric
    x = np.arange(len(subjects))
    width = 0.2

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = width * (i - 1.5)
        bars = ax_bar.bar(x + offset, data[metric], width, label=label)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7)

    ax_bar.set_xlabel('Subject ID', fontsize=11, fontweight='bold')
    ax_bar.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax_bar.set_title('Validation Metrics by Subject', fontsize=13, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(subjects)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis='y', alpha=0.3)

    # Bottom: Four line graphs (2x2)
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row = 1 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        values = [results[subject]['val_metrics'][metric] for subject in subjects]
        x_line = np.arange(len(subjects))

        ax.plot(x_line, values, marker='o', linewidth=2, markersize=6, label=label, color=f'C{idx}')

        # Add value labels
        for i, (xi, yi) in enumerate(zip(x_line, values)):
            ax.text(xi, yi, f'{yi:.2f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Subject ID', fontsize=9, fontweight='bold')
        ax.set_ylabel(label, fontsize=9, fontweight='bold')
        ax.set_title(f'{label} Trend', fontsize=10, fontweight='bold')
        ax.set_xticks(x_line)
        ax.set_xticklabels(subjects, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'μ={mean_val:.3f}')
        ax.legend(fontsize=7)

    plt.suptitle('Leave-One-Subject-Out Cross-Validation Results',
                 fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize leave-one-out experiment results')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualizations (defaults to results_dir)')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found! Make sure to run the experiment first.")
        return

    print(f"Loaded results for {len(results)} subjects: {sorted(results.keys())}")

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    print("\nCreating visualizations...")

    # Bar graph
    bar_graph_path = output_dir / "metrics_bar_graph.png"
    create_bar_graph(results, bar_graph_path)

    # Line graphs
    line_graphs_path = output_dir / "metrics_line_graphs.png"
    create_line_graphs(results, line_graphs_path)

    # Combined visualization
    combined_path = output_dir / "metrics_combined.png"
    create_combined_visualization(results, combined_path)

    # Print summary statistics
    print_summary_statistics(results)

    print(f"\nVisualization complete! Check {output_dir} for results.")


if __name__ == "__main__":
    main()
