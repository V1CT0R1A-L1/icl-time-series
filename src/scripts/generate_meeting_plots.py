"""
Script to generate visualizations for meeting reports.
Creates plots for:
1. Task structure visualization
2. Learning curves (if wandb data exists)
3. Comparison across configurations
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Learning curves will not be generated.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Task visualization will be limited.")


sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


def plot_task_structure(output_dir="meeting_plots"):
    """Visualize the task structure: context clusters and target cluster."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Example visualization: show the sequence structure
    K = 2  # Number of components
    C = 8  # Points per context cluster
    T_target = 5  # Context points in target cluster
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Draw context clusters
    context_colors = ['#3498db', '#e74c3c']  # Blue and red
    for k in range(K):
        x_start = k * C
        x_end = (k + 1) * C
        ax.axvspan(x_start, x_end, alpha=0.2, color=context_colors[k], 
                  label=f'Context Cluster {k+1}' if k == 0 else None)
        # Add cluster label
        ax.text((x_start + x_end) / 2, 0.5, f'Cluster {k+1}\n(C={C})', 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw target cluster
    target_start = K * C
    target_context_end = target_start + T_target
    predict_idx = target_start + T_target
    
    ax.axvspan(target_start, target_context_end, alpha=0.2, color='#2ecc71', 
              label='Target Cluster Context')
    ax.axvline(predict_idx, color='black', linestyle='--', linewidth=2, 
              label='Prediction Point')
    
    # Add labels
    ax.text((target_start + target_context_end) / 2, 0.5, 
           f'Target Context\n(T={T_target})', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(predict_idx, 0.8, 'PREDICT', ha='center', fontsize=12, 
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Set up axes
    total_points = K * C + T_target + 1
    ax.set_xlim(-1, total_points + 1)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Sequence Position', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_title('Grouped Mixture Linear Regression Task Structure', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticks(range(0, total_points + 1, 4))
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])
    
    # Add description
    desc = (f"Structure: {K} context clusters × {C} points each = {K*C} context points\n"
            f"Target cluster: {T_target} context points + 1 prediction = {T_target+1} target points\n"
            f"Total: {total_points} points")
    ax.text(0.5, -0.3, desc, transform=ax.transAxes, ha='center', 
           fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "task_structure.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved task structure plot: {output_path}")
    plt.close()


def plot_learning_curves(run_names=None, output_dir="meeting_plots", project="in-context-learning"):
    """Plot learning curves from wandb runs."""
    if not WANDB_AVAILABLE:
        print("⚠ Skipping learning curves (wandb not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Get runs
    runs = api.runs(f"{project}")
    
    if run_names is None:
        # Look for group_mixture_linear runs
        runs = [r for r in runs if "mixture" in r.name.lower() or "group" in r.name.lower()]
    else:
        runs = [r for r in runs if any(name in r.name for name in run_names)]
    
    if not runs:
        print("⚠ No matching wandb runs found. Skipping learning curves.")
        return
    
    # Plot learning curves
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Overall loss
    ax1 = axes[0]
    for run in runs:
        history = run.history()
        if 'overall_loss' in history.columns:
            steps = history['_step'].values
            losses = history['overall_loss'].values
            ax1.plot(steps, losses, label=run.name, linewidth=2)
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Excess loss
    ax2 = axes[1]
    for run in runs:
        history = run.history()
        if 'excess_loss' in history.columns:
            steps = history['_step'].values
            excess = history['excess_loss'].values
            ax2.plot(steps, excess, label=run.name, linewidth=2)
    
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Excess Loss (vs baseline)', fontsize=12)
    ax2.set_title('Excess Loss', fontsize=14, fontweight='bold')
    ax2.axhline(1.0, color='gray', linestyle='--', label='Baseline')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved learning curves: {output_path}")
    plt.close()


def plot_config_comparison(output_dir="meeting_plots"):
    """Create a comparison plot showing different configurations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define configurations to compare
    configs = [
        {"name": "1 Component\n(C=10, T=3)", "K": 1, "C": 10, "T": 3, "color": palette[0]},
        {"name": "2 Components\n(C=8, T=5)", "K": 2, "C": 8, "T": 5, "color": palette[1]},
        {"name": "3 Components\n(C=6, T=7)", "K": 3, "C": 6, "T": 7, "color": palette[2]},
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x_pos = np.arange(len(configs))
    context_sizes = [c["K"] * c["C"] for c in configs]
    total_sizes = [c["K"] * c["C"] + c["T"] + 1 for c in configs]
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, context_sizes, width, label='Context Size', 
                   color=[c["color"] for c in configs], alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, total_sizes, width, label='Total Sequence Length',
                   color=[c["color"] for c in configs], alpha=0.9)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Sequence Length', fontsize=12)
    ax.set_title('Task Complexity Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c["name"] for c in configs])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add complexity indicators
    for i, c in enumerate(configs):
        complexity = c["K"] * c["C"] + c["T"] + 1
        ax.text(i, complexity + 2, f'K={c["K"]}', ha='center', fontsize=9, 
               style='italic')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "config_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved config comparison: {output_path}")
    plt.close()


def create_summary_plot(output_dir="meeting_plots"):
    """Create a single-page summary with all key visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Task Structure (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    K, C, T_target = 2, 8, 5
    context_colors = ['#3498db', '#e74c3c']
    
    for k in range(K):
        x_start = k * C
        x_end = (k + 1) * C
        ax1.axvspan(x_start, x_end, alpha=0.2, color=context_colors[k],
                   label=f'Context Cluster {k+1}' if k == 0 else None)
        ax1.text((x_start + x_end) / 2, 0.5, f'C{k+1}',
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    target_start = K * C
    target_context_end = target_start + T_target
    predict_idx = target_start + T_target
    ax1.axvspan(target_start, target_context_end, alpha=0.2, color='#2ecc71',
               label='Target Context')
    ax1.axvline(predict_idx, color='black', linestyle='--', linewidth=2,
               label='Predict')
    ax1.text((target_start + target_context_end) / 2, 0.5, f'T={T_target}',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(predict_idx, 0.8, '?', ha='center', fontsize=14, fontweight='bold')
    ax1.set_xlim(-1, predict_idx + 2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_xlabel('Sequence Position', fontsize=11)
    ax1.set_title('Task Structure: Grouped Mixture Linear Regression', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks([])
    
    # 2. Config Comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    configs = [
        {"name": "K=1", "size": 14},
        {"name": "K=2", "size": 22},
        {"name": "K=3", "size": 30},
    ]
    x_pos = np.arange(len(configs))
    sizes = [c["size"] for c in configs]
    bars = ax2.bar(x_pos, sizes, color=palette[:3], alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c["name"] for c in configs])
    ax2.set_ylabel('Sequence Length', fontsize=10)
    ax2.set_title('Task Complexity', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 3. Key Findings (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    findings = [
        "✓ Random cluster order removes\n  positional cues",
        "✓ All components guaranteed\n  in context",
        "✓ Model must infer component\n  from target context",
        "✓ Evaluation on prediction\n  point only",
    ]
    y_pos = np.linspace(0.9, 0.1, len(findings))
    for finding, y in zip(findings, y_pos):
        ax3.text(0.1, y, finding, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax3.set_title('Key Design Choices', fontsize=11, fontweight='bold')
    
    # 4. Next Steps (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    steps = [
        "1. Ablation studies\n   (cluster order, T, K)",
        "2. Scale sensitivity\n   analysis",
        "3. Curriculum learning\n   experiments",
        "4. Generalization to\n   held-out configurations",
    ]
    y_pos = np.linspace(0.9, 0.1, len(steps))
    for step, y in zip(steps, y_pos):
        ax4.text(0.1, y, step, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax4.set_title('Next Steps', fontsize=11, fontweight='bold')
    
    # 5. Task Description (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    desc = (
        "Task: Predict y for a new x\n"
        "given K context clusters\n"
        "and target context.\n\n"
        "Challenge: Infer which\n"
        "component the target\n"
        "cluster uses, then predict.\n\n"
        "Baseline: Random guessing\n"
        "or average of contexts."
    )
    ax5.text(0.5, 0.5, desc, fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.set_title('Task Description', fontsize=11, fontweight='bold')
    
    plt.suptitle('In-Context Learning: Grouped Mixture Linear Regression', 
                fontsize=14, fontweight='bold', y=0.98)
    
    output_path = os.path.join(output_dir, "meeting_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for meeting report")
    parser.add_argument("--output-dir", type=str, default="meeting_plots",
                       help="Output directory for plots")
    parser.add_argument("--wandb-project", type=str, default="in-context-learning",
                       help="Wandb project name")
    parser.add_argument("--run-names", type=str, nargs='+', default=None,
                       help="Specific wandb run names to plot")
    parser.add_argument("--skip-wandb", action="store_true",
                       help="Skip wandb learning curves")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Meeting Plots")
    print("=" * 60)
    
    # 1. Task structure
    print("\n1. Generating task structure plot...")
    plot_task_structure(args.output_dir)
    
    # 2. Learning curves (if wandb available)
    if not args.skip_wandb:
        print("\n2. Generating learning curves...")
        plot_learning_curves(args.run_names, args.output_dir, args.wandb_project)
    else:
        print("\n2. Skipping learning curves (--skip-wandb)")
    
    # 3. Config comparison
    print("\n3. Generating config comparison...")
    plot_config_comparison(args.output_dir)
    
    # 4. Summary plot
    print("\n4. Generating summary plot...")
    create_summary_plot(args.output_dir)
    
    print("\n" + "=" * 60)
    print(f"✓ All plots saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
