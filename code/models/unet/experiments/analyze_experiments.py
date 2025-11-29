#!/usr/bin/env python3
"""
Analyze all experiments and create scientific visualizations for the report.
"""

import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.style as style

# Use a scientific plotting style
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def parse_output_file(output_path):
    """Parse output.txt file and extract metrics."""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'final_train_loss': None,
        'final_train_accuracy': None,
        'final_train_precision': None,
        'final_train_recall': None,
        'best_val_loss': None,
        'best_val_accuracy': None,
        'best_val_precision': None,
        'best_val_recall': None,
    }
    
    with open(output_path, 'r') as f:
        content = f.read()
    
    # Pattern for epoch metrics - matches "Epoch X Train" or "Epoch X/Y Train"
    epoch_pattern = re.compile(
        r'Epoch (\d+)(?:/(\d+))? Train - Loss: ([\d.]+), Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+)'
    )
    val_pattern = re.compile(
        r'Epoch (\d+)(?:/(\d+))? Val\s+ - Loss: ([\d.]+), Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+)'
    )
    
    # Extract epoch-by-epoch metrics
    for match in epoch_pattern.finditer(content):
        epoch = int(match.group(1))
        metrics['epochs'].append(epoch)
        metrics['train_loss'].append(float(match.group(3)))
        metrics['train_accuracy'].append(float(match.group(4)))
        metrics['train_precision'].append(float(match.group(5)))
        metrics['train_recall'].append(float(match.group(6)))
    
    for match in val_pattern.finditer(content):
        epoch = int(match.group(1))
        metrics['val_loss'].append(float(match.group(3)))
        metrics['val_accuracy'].append(float(match.group(4)))
        metrics['val_precision'].append(float(match.group(5)))
        metrics['val_recall'].append(float(match.group(6)))
    
    # Extract final metrics - handle both "Best Validation Metrics:" and "\Best Validation Metrics:"
    final_train_match = re.search(
        r'Final Training Metrics:.*?Loss:\s+([\d.]+).*?Accuracy:\s+([\d.]+).*?Precision:\s+([\d.]+).*?Recall:\s+([\d.]+)',
        content, re.DOTALL
    )
    if final_train_match:
        metrics['final_train_loss'] = float(final_train_match.group(1))
        metrics['final_train_accuracy'] = float(final_train_match.group(2))
        metrics['final_train_precision'] = float(final_train_match.group(3))
        metrics['final_train_recall'] = float(final_train_match.group(4))
    
    # Try both patterns for validation metrics (Best or Final)
    best_val_match = re.search(
        r'[\\]?(?:Best|Final) Validation Metrics:.*?Loss:\s+([\d.]+).*?Accuracy:\s+([\d.]+).*?Precision:\s+([\d.]+).*?Recall:\s+([\d.]+)',
        content, re.DOTALL
    )
    if best_val_match:
        metrics['best_val_loss'] = float(best_val_match.group(1))
        metrics['best_val_accuracy'] = float(best_val_match.group(2))
        metrics['best_val_precision'] = float(best_val_match.group(3))
        metrics['best_val_recall'] = float(best_val_match.group(4))
    
    return metrics

def get_experiment_name(exp_num, config):
    """Generate a descriptive name for the experiment."""
    augs = config.get('augmentations', {})
    in_place = augs.get('in_place_augmentations', [])
    enrichment = augs.get('enrichment_augmentations', [])
    epochs = config.get('training', {}).get('epochs', 0)
    
    name_parts = [f"Exp {exp_num}"]
    
    # Add augmentation info
    if 'speckle' in in_place and 'gamma' in in_place:
        name_parts.append("+Speckle+Gamma")
    elif 'speckle' in in_place:
        name_parts.append("+Speckle")
    elif 'gamma' in in_place:
        name_parts.append("+Gamma")
    
    if enrichment:
        if 'random_resize_crop' in enrichment and 'flip' in enrichment:
            name_parts.append("+Crop+Flip")
        elif 'random_resize_crop' in enrichment:
            crop_pct = augs.get('random_resize_crop_percent', 20)
            name_parts.append(f"+Crop({crop_pct}%)")
        elif 'flip' in enrichment:
            name_parts.append("+Flip")
    else:
        name_parts.append("NoEnrich")
    
    name_parts.append(f"({epochs}ep)")
    
    return " ".join(name_parts)

def load_all_experiments(base_dir):
    """Load all experiments."""
    experiments = {}
    base_path = Path(base_dir)
    
    for exp_dir in sorted(base_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_num = exp_dir.name
        config_path = exp_dir / 'config.json'
        output_path = exp_dir / 'output.txt'
        
        if not config_path.exists() or not output_path.exists():
            print(f"Warning: Missing files for experiment {exp_num}")
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            metrics = parse_output_file(output_path)
            exp_name = get_experiment_name(exp_num, config)
            
            experiments[exp_num] = {
                'name': exp_name,
                'config': config,
                'metrics': metrics
            }
            
            print(f"Loaded {exp_name}: {len(metrics['epochs'])} epochs")
        except Exception as e:
            print(f"Error loading experiment {exp_num}: {e}")
    
    return experiments

def plot_learning_curves(experiments, output_dir):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Learning Curves Across Experiments', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for (exp_num, exp_data), color in zip(experiments.items(), colors):
        metrics = exp_data['metrics']
        if metrics['epochs'] and metrics['train_loss']:
            ax.plot(metrics['epochs'], metrics['train_loss'], 
                   label=exp_data['name'], color=color, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Over Epochs')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for (exp_num, exp_data), color in zip(experiments.items(), colors):
        metrics = exp_data['metrics']
        if metrics['epochs'] and metrics['val_loss']:
            ax.plot(metrics['epochs'], metrics['val_loss'], 
                   label=exp_data['name'], color=color, linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Over Epochs')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax = axes[1, 0]
    for (exp_num, exp_data), color in zip(experiments.items(), colors):
        metrics = exp_data['metrics']
        if metrics['epochs'] and metrics['train_accuracy']:
            ax.plot(metrics['epochs'], metrics['train_accuracy'], 
                   label=exp_data['name'], color=color, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.set_title('Training Accuracy Over Epochs')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.8, 1.0])
    
    # Plot 4: Validation Accuracy
    ax = axes[1, 1]
    for (exp_num, exp_data), color in zip(experiments.items(), colors):
        metrics = exp_data['metrics']
        if metrics['epochs'] and metrics['val_accuracy']:
            ax.plot(metrics['epochs'], metrics['val_accuracy'], 
                   label=exp_data['name'], color=color, linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Over Epochs')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.6, 0.8])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'learning_curves.png'}")
    plt.close()

def plot_final_metrics_comparison(experiments, output_dir):
    """Plot bar chart comparing final metrics across experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Final Metrics Comparison Across Experiments', fontsize=16, fontweight='bold')
    
    exp_names = [exp_data['name'] for exp_data in experiments.values()]
    x_pos = np.arange(len(exp_names))
    width = 0.35
    
    # Extract metrics - handle None values
    train_acc = [exp_data['metrics']['final_train_accuracy'] or 0 for exp_data in experiments.values()]
    val_acc = [exp_data['metrics']['best_val_accuracy'] or 0 for exp_data in experiments.values()]
    train_prec = [exp_data['metrics']['final_train_precision'] or 0 for exp_data in experiments.values()]
    val_prec = [exp_data['metrics']['best_val_precision'] or 0 for exp_data in experiments.values()]
    train_rec = [exp_data['metrics']['final_train_recall'] or 0 for exp_data in experiments.values()]
    val_rec = [exp_data['metrics']['best_val_recall'] or 0 for exp_data in experiments.values()]
    train_loss = [exp_data['metrics']['final_train_loss'] or 0 for exp_data in experiments.values()]
    val_loss = [exp_data['metrics']['best_val_loss'] or 0 for exp_data in experiments.values()]
    
    # Plot 1: Accuracy
    ax = axes[0, 0]
    bars1 = ax.bar(x_pos - width/2, train_acc, width, label='Training', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, val_acc, width, label='Validation', alpha=0.8)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Accuracy Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.6, 1.0])
    
    # Plot 2: Precision
    ax = axes[0, 1]
    bars1 = ax.bar(x_pos - width/2, train_prec, width, label='Training', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, val_prec, width, label='Validation', alpha=0.8)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Precision')
    ax.set_title('Final Precision Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.0])
    
    # Plot 3: Recall
    ax = axes[1, 0]
    bars1 = ax.bar(x_pos - width/2, train_rec, width, label='Training', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, val_rec, width, label='Validation', alpha=0.8)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Recall')
    ax.set_title('Final Recall Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.6, 1.0])
    
    # Plot 4: Loss
    ax = axes[1, 1]
    bars1 = ax.bar(x_pos - width/2, train_loss, width, label='Training', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, val_loss, width, label='Validation', alpha=0.8)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Loss')
    ax.set_title('Final Loss Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'final_metrics_comparison.png'}")
    plt.close()

def plot_precision_recall_comparison(experiments, output_dir):
    """Plot precision-recall comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for (exp_num, exp_data), color in zip(experiments.items(), colors):
        metrics = exp_data['metrics']
        train_prec = metrics['final_train_precision']
        train_rec = metrics['final_train_recall']
        val_prec = metrics['best_val_precision']
        val_rec = metrics['best_val_recall']
        
        if train_prec and train_rec:
            ax.scatter(train_rec, train_prec, s=150, color=color, marker='o', 
                      label=f"{exp_data['name']} (Train)", alpha=0.7, edgecolors='black', linewidth=1.5)
        if val_prec and val_rec:
            ax.scatter(val_rec, val_prec, s=150, color=color, marker='s', 
                      label=f"{exp_data['name']} (Val)", alpha=0.7, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'precision_recall_comparison.png'}")
    plt.close()

def create_summary_table(experiments, output_dir):
    """Create a summary table of all experiments."""
    import pandas as pd
    
    data = []
    for exp_num, exp_data in experiments.items():
        metrics = exp_data['metrics']
        data.append({
            'Experiment': exp_data['name'],
            'Train Loss': f"{metrics['final_train_loss']:.4f}" if metrics['final_train_loss'] else "N/A",
            'Train Acc': f"{metrics['final_train_accuracy']:.4f}" if metrics['final_train_accuracy'] else "N/A",
            'Train Prec': f"{metrics['final_train_precision']:.4f}" if metrics['final_train_precision'] else "N/A",
            'Train Rec': f"{metrics['final_train_recall']:.4f}" if metrics['final_train_recall'] else "N/A",
            'Val Loss': f"{metrics['best_val_loss']:.4f}" if metrics['best_val_loss'] else "N/A",
            'Val Acc': f"{metrics['best_val_accuracy']:.4f}" if metrics['best_val_accuracy'] else "N/A",
            'Val Prec': f"{metrics['best_val_precision']:.4f}" if metrics['best_val_precision'] else "N/A",
            'Val Rec': f"{metrics['best_val_recall']:.4f}" if metrics['best_val_recall'] else "N/A",
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'experiments_summary.csv', index=False)
    print(f"Saved: {output_dir / 'experiments_summary.csv'}")
    
    # Also create a LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.4f", escape=False)
    with open(output_dir / 'experiments_summary.tex', 'w') as f:
        f.write(latex_table)
    print(f"Saved: {output_dir / 'experiments_summary.tex'}")

def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading experiments...")
    experiments = load_all_experiments(base_dir)
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"\nLoaded {len(experiments)} experiments")
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_learning_curves(experiments, output_dir)
    plot_final_metrics_comparison(experiments, output_dir)
    plot_precision_recall_comparison(experiments, output_dir)
    create_summary_table(experiments, output_dir)
    
    print("\nAll visualizations generated successfully!")
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()

