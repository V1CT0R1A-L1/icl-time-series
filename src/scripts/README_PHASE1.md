# Phase 1: Component Count Scaling - Automation Guide

## Overview

This automation system runs systematic experiments to investigate how transformer models scale with the number of components (K) in mixture linear regression tasks.

## Quick Start

### 1. Generate Configs Only (Preview)
```bash
python src/scripts/phase1_sweep.py
```

This generates config files for K = [1, 2, 3, 4, 5, 6, 8, 10] without running experiments.

### 2. Run Experiments Locally
```bash
# Run all experiments sequentially
python src/scripts/phase1_sweep.py --run

# Run specific range
python src/scripts/phase1_sweep.py --run --start-from 2 --stop-at 5

# Dry run (see what would be executed)
python src/scripts/phase1_sweep.py --dry-run
```

### 3. Run on HPC (MIT Engaging)
```bash
# Make script executable
chmod +x src/scripts/submit_phase1_hpc.sh

# Submit all jobs
./src/scripts/submit_phase1_hpc.sh

# Submit specific range
./src/scripts/submit_phase1_hpc.sh 2 10
```

### 4. Analyze Results
```bash
# Using wandb
python src/scripts/analyze_phase1.py --wandb-project "in-context-learning" --k-values 1 2 3 4 5 6 8 10

# Using local results
python src/scripts/analyze_phase1.py --results-dir ../models --k-values 1 2 3 4 5 6 8 10
```

## Experiment Parameters

- **K values**: [1, 2, 3, 4, 5, 6, 8, 10]
- **Total contexts**: 24 (constant across all K)
- **Contexts per component**: 24 / K (rounded)
- **Target cluster context points**: 3 (constant)
- **Scale**: 0.1 (small for easier learning)
- **Training steps**: 5001

## Generated Files

- **Configs**: `src/conf/phase1/k{1,2,3,...}.yaml`
- **Models**: `../models/phase1_k{1,2,3,...}/`
- **Plots**: `results/phase1/*.png` and `*.pdf`

## Research Questions

1. **Does loss scale with K?** → `loss_vs_components.png`
2. **Is there a hard capacity limit?** → `capacity_analysis.png`
3. **How does required training steps grow?** → `convergence_steps.png`

## Troubleshooting

- **Config not found**: Run `python src/scripts/phase1_sweep.py` first to generate configs
- **HPC jobs fail**: Check `logs/phase1_k*_*.err` for error messages
- **No wandb data**: Ensure wandb is properly configured in your configs

## Next Steps

After Phase 1 completes, proceed to:
- Phase 2: Sample Efficiency (vary contexts_per_component)
- Phase 3: Dimensionality Scaling (vary n_dims)
- Phase 4: Noise Robustness (vary noise_std)
