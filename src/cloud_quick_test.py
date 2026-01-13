#!/usr/bin/env python3
"""
Quick test to run on the cloud to verify everything works
"""
import sys
import os

def main():
    print("🔍 Cloud Quick Test - Checking Basic Functionality")
    print("=" * 60)
    
    # Test 1: Basic imports
    try:
        import torch
        import yaml
        from transformers import GPT2Model, GPT2Config
        print("✓ Core imports successful")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return 1
    
    # Test 2: Our custom imports
    try:
        sys.path.append(os.getcwd())  # Add current directory to path
        
        from tasks import MultiContextMixtureLinear, get_task_sampler
        from samplers import MultiContextMixtureSampler, get_data_sampler  
        from models import TransformerModel
        from curriculum import Curriculum
        print("✓ Custom module imports successful")
    except ImportError as e:
        print(f"❌ Custom import failed: {e}")
        return 1
    
    # Test 3: Basic data generation
    try:
        print("\n🧪 Testing data generation...")
        n_dims = 8
        batch_size = 2
        
        task = MultiContextMixtureLinear(
            n_dims=n_dims, batch_size=batch_size,
            n_contexts=2, context_length=4, predict_length=1
        )
        
        sampler = MultiContextMixtureSampler(
            n_dims=n_dims, n_contexts=2, context_length=4, predict_length=1
        )
        
        structure = sampler.get_sequence_structure()
        xs = sampler.sample_xs(structure['total_length'], batch_size)
        ys = task.evaluate(xs)
        
        print(f"✓ Data generation: xs {xs.shape}, ys {ys.shape}")
        print(f"✓ Sequence structure: {len(structure['sep_positions'])} SEP tokens")
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return 1
    
    # Test 4: Model instantiation
    try:
        print("\n🤖 Testing model...")
        model = TransformerModel(
            n_dims=n_dims, n_positions=50,
            n_embd=64, n_layer=2, n_head=2
        )
        print(f"✓ Model created: {model.name}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(xs, ys)
            print(f"✓ Forward pass: output {output.shape}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return 1
    
    # Test 5: Training step simulation
    try:
        print("\n⚡ Testing training step...")
        import torch.nn as nn
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        
        # Simple train step
        optimizer.zero_grad()
        output = model(xs, ys)
        loss = loss_fn(output, ys)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step completed: loss {loss.item():.6f}")
        print(f"✓ Gradients computed: {any(p.grad is not None for p in model.parameters())}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 ALL CLOUD TESTS PASSED! Ready for training! 🎉")
    return 0

if __name__ == "__main__":
    sys.exit(main())