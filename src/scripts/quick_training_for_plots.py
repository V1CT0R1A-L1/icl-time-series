"""
Quick training script to generate data for meeting plots.
Runs a short training session with minimal configuration.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def create_quick_config(output_dir, steps=500):
    """Create a minimal config for quick training."""
    config_content = f"""inherit:
  - models/standard.yaml
  - wandb.yaml

model:
  n_dims: 5
  n_positions: 30

training:
  task: group_mixture_linear
  data: group_mixture_linear
  task_kwargs:
    n_components: 1
    contexts_per_component: 10
    target_cluster_context_points: 3
    noise_std: 0.0
    scale: 0.1
  batch_size: 64
  learning_rate: 0.0005
  save_every_steps: {steps // 2}
  keep_every_steps: 100000
  train_steps: {steps}
  curriculum:
    dims:
      start: 5
      end: 5
      inc: 1
      interval: {steps + 1}
    points:
      start: 14
      end: 14
      inc: 2
      interval: {steps + 1}

out_dir: ../models/quick_training

wandb:
  name: "quick_training_for_plots"

test_run: False
"""
    
    config_dir = Path(__file__).parent.parent / "conf"
    config_path = config_dir / "quick_training_for_plots.yaml"
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Quick training for plot generation")
    parser.add_argument("--steps", type=int, default=500,
                       help="Number of training steps")
    parser.add_argument("--config", type=str, default=None,
                       help="Use existing config file (overrides --steps)")
    parser.add_argument("--n-components", type=int, default=1,
                       help="Number of mixture components")
    parser.add_argument("--contexts-per-component", type=int, default=10,
                       help="Points per context cluster")
    parser.add_argument("--target-context-points", type=int, default=3,
                       help="Context points in target cluster")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Quick Training for Plot Generation")
    print("=" * 60)
    
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return
    else:
        print(f"\nCreating quick training config ({args.steps} steps)...")
        config_path = create_quick_config("../models/quick_training", args.steps)
        print(f"✓ Config created: {config_path}")
    
    # Run train.py as a subprocess
    src_dir = Path(__file__).parent.parent
    train_script = src_dir / "train.py"
    
    # Use path relative to src directory (train.py expects paths relative to cwd or absolute)
    # Since we'll run from src/, use relative path if config is in src/conf/
    if config_path.parent.name == "conf" and str(src_dir) in str(config_path.parent):
        config_arg = f"conf/{config_path.name}"
    else:
        # Use absolute path
        config_arg = str(config_path)
    
    cmd = [sys.executable, str(train_script), "--config", config_arg]
    
    print(f"\nStarting training with config: {config_path}")
    print("-" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=str(src_dir))
        if result.returncode == 0:
            print("\n✓ Training completed!")
            print(f"  Check wandb for learning curves")
            print(f"  Run generate_meeting_plots.py to create visualizations")
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
