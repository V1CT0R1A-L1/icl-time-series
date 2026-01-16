#!/usr/bin/env python3
"""
Analysis script for Phase 1: Component Count Scaling
Generates publication-quality visualizations from experiment results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import wandb
import argparse

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_wandb_data(project_name, k_values):
    """Load data from wandb runs."""
    api = wandb.Api()
    runs = api.runs(project_name)
    
    data = []
    for run in runs:
        if run.name.startswith("phase1_k"):
            try:
                k = int(run.name.split("_k")[1])
                if k in k_values:
                    history = run.history()
                    if len(history) > 0:
                        final_loss = history['overall_loss'].iloc[-1]
                        final_step = history['_step'].iloc[-1]
                        data.append({
                            'K': k,
                            'final_loss': final_loss,
                            'final_step': final_step,
                            'min_loss': history['overall_loss'].min(),
                            'convergence_step': history[history['overall_loss'] < 0.01]['_step'].min() if (history['overall_loss'] < 0.01).any() else None
                        })
            except:
                continue
    
    return pd.DataFrame(data)

def load_local_data(results_dir):
    """Load data from local result files."""
    data = []
    results_path = Path(results_dir)
    
    for k_dir in results_path.glob("phase1_k*"):
        try:
            k = int(k_dir.name.split("_k")[1])
            # Look for metrics files or logs
            # This is a placeholder - adjust based on your logging format
            metrics_file = k_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    data.append({
                        'K': k,
                        'final_loss': metrics.get('final_loss'),
                        'min_loss': metrics.get('min_loss'),
                    })
        except:
            continue
    
    return pd.DataFrame(data)

def plot_loss_vs_components(df, output_dir):
    """Plot 1: Final loss vs number of components."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(df['K'], df['final_loss'], 'o-', linewidth=2, markersize=8, label='Final Loss')
    ax.plot(df['K'], df['min_loss'], 's--', linewidth=2, markersize=8, label='Minimum Loss', alpha=0.7)
    
    ax.set_xlabel('Number of Components (K)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Scaling with Component Count', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_vs_components.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'loss_vs_components.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'loss_vs_components.png'}")
    plt.close()

def plot_convergence_steps(df, output_dir):
    """Plot 2: Training steps to convergence vs K."""
    if 'convergence_step' not in df.columns or df['convergence_step'].isna().all():
        print("No convergence step data available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(df['K'], df['convergence_step'], 'o-', linewidth=2, markersize=8, color='green')
    
    ax.set_xlabel('Number of Components (K)', fontsize=12)
    ax.set_ylabel('Steps to Convergence (< 0.01 loss)', fontsize=12)
    ax.set_title('Training Steps Required vs Component Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_steps.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'convergence_steps.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'convergence_steps.png'}")
    plt.close()

def plot_capacity_analysis(df, output_dir):
    """Plot 3: Capacity limit analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss scaling
    ax1.plot(df['K'], df['final_loss'], 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=0.1, color='r', linestyle='--', label='Threshold (0.1)')
    ax1.set_xlabel('Number of Components (K)', fontsize=12)
    ax1.set_ylabel('Final Loss', fontsize=12)
    ax1.set_title('Loss vs Components', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Loss per component
    df['loss_per_component'] = df['final_loss'] / df['K']
    ax2.plot(df['K'], df['loss_per_component'], 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Components (K)', fontsize=12)
    ax2.set_ylabel('Loss per Component', fontsize=12)
    ax2.set_title('Loss per Component vs K', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'capacity_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'capacity_analysis.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'capacity_analysis.png'}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 1 results")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name (if using wandb)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../models",
        help="Directory with local results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase1",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 8, 10],
        help="K values to analyze"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if args.wandb_project:
        print(f"Loading data from wandb project: {args.wandb_project}")
        df = load_wandb_data(args.wandb_project, args.k_values)
    else:
        print(f"Loading data from local directory: {args.results_dir}")
        df = load_local_data(args.results_dir)
    
    if df.empty:
        print("No data found! Check your data source.")
        return
    
    df = df.sort_values('K')
    print(f"\nLoaded data for {len(df)} experiments:")
    print(df.to_string())
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_loss_vs_components(df, output_dir)
    plot_convergence_steps(df, output_dir)
    plot_capacity_analysis(df, output_dir)
    
    # Save data
    df.to_csv(output_dir / 'phase1_results.csv', index=False)
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
