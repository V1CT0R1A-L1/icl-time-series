#!/usr/bin/env python3
"""
Phase 1: Component Count Scaling Experiment
Sweeps over K (number of components) and generates configs/runs experiments.

Research Questions:
1. Does loss scale with K?
2. Is there a hard capacity limit?
3. How does required training steps grow?
"""

import yaml
import subprocess
import os
import sys
import argparse
from pathlib import Path
import time

# Experiment parameters
K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10]
TOTAL_CONTEXTS = 24
TARGET_CLUSTER_CONTEXT_POINTS = 3
BASE_CONFIG = "src/conf/phase1_component_scaling_template.yaml"
OUTPUT_DIR = "src/conf/phase1"

def calculate_contexts_per_component(k, total_contexts):
    """Calculate contexts per component, rounding to nearest integer."""
    return round(total_contexts / k)

def calculate_total_length(k, contexts_per_component, target_context_points):
    """Calculate total sequence length."""
    return (k * contexts_per_component) + target_context_points + 1

def generate_config(k, contexts_per_component, total_length):
    """Generate a config file for a specific K value."""
    with open(BASE_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update K-specific values
    config['training']['task_kwargs']['n_components'] = k
    config['training']['task_kwargs']['contexts_per_component'] = contexts_per_component
    config['training']['curriculum']['points']['start'] = total_length
    config['training']['curriculum']['points']['end'] = total_length
    
    # Update n_positions (need at least 2 * total_length)
    config['model']['n_positions'] = max(60, 2 * total_length + 10)
    
    # Update wandb name
    config['wandb']['name'] = f"phase1_k{k}"
    
    # Update output dir
    config['out_dir'] = f"../models/phase1_k{k}"
    
    # Save config
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_path = f"{OUTPUT_DIR}/k{k}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return config_path

def run_experiment(config_path, k, dry_run=False):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running experiment for K={k}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")
    
    if dry_run:
        print(f"[DRY RUN] Would execute: python src/train.py --config {config_path}")
        return True
    
    cmd = [sys.executable, "src/train.py", "--config", config_path]
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ K={k} completed successfully in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ K={k} failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ K={k} interrupted by user")
        return False

def main():
    """Generate configs and optionally run experiments."""
    parser = argparse.ArgumentParser(description="Phase 1: Component Count Scaling")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run the experiments (default: just generate configs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=None,
        help="Start from this K value (useful for resuming)"
    )
    parser.add_argument(
        "--stop-at",
        type=int,
        default=None,
        help="Stop at this K value"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 1: Component Count Scaling")
    print("="*60)
    
    # Generate configs
    configs = []
    for k in K_VALUES:
        # Apply filters
        if args.start_from and k < args.start_from:
            continue
        if args.stop_at and k > args.stop_at:
            break
            
        c = calculate_contexts_per_component(k, TOTAL_CONTEXTS)
        total_len = calculate_total_length(k, c, TARGET_CLUSTER_CONTEXT_POINTS)
        
        print(f"\nK={k}:")
        print(f"  Contexts per component: {c} (target: {TOTAL_CONTEXTS/k:.2f})")
        print(f"  Total sequence length: {total_len}")
        print(f"  Total context points: {k * c} (target: {TOTAL_CONTEXTS})")
        
        config_path = generate_config(k, c, total_len)
        configs.append((k, config_path))
        print(f"  Config saved: {config_path}")
    
    print("\n" + "="*60)
    print(f"Generated {len(configs)} configs")
    print("="*60)
    
    # Run experiments if requested
    if args.run or args.dry_run:
        print(f"\n{'='*60}")
        print(f"Running experiments ({'DRY RUN' if args.dry_run else 'LIVE'})")
        print(f"{'='*60}")
        
        results = {}
        for k, config_path in configs:
            success = run_experiment(config_path, k, dry_run=args.dry_run)
            results[k] = success
            
            if not success and not args.dry_run:
                response = input(f"\nK={k} failed. Continue? (y/n): ")
                if response.lower() != 'y':
                    print("Stopping experiment sweep.")
                    break
        
        # Summary
        print("\n" + "="*60)
        print("Experiment Summary")
        print("="*60)
        for k, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} K={k}")
        
        successful = sum(1 for s in results.values() if s)
        print(f"\nCompleted: {successful}/{len(results)} experiments")
    else:
        print("\n" + "="*60)
        print("To run experiments, use:")
        print("="*60)
        print("  python src/scripts/phase1_sweep.py --run")
        print("\nOr run individually:")
        for k, path in configs:
            print(f"  python src/train.py --config {path}")

if __name__ == "__main__":
    main()
