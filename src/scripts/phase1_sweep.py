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
from pathlib import Path

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
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config_path

def main():
    """Generate configs and optionally run experiments."""
    print("="*60)
    print("Phase 1: Component Count Scaling - Config Generation")
    print("="*60)
    
    configs = []
    for k in K_VALUES:
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
    print("Generated configs:")
    for k, path in configs:
        print(f"  K={k}: {path}")
    
    print("\n" + "="*60)
    print("To run experiments, use:")
    for k, path in configs:
        print(f"  python src/train.py --config {path}")
    print("\nOr run all sequentially:")
    for k, path in configs:
        print(f"  python src/train.py --config {path} && \\")

if __name__ == "__main__":
    main()
