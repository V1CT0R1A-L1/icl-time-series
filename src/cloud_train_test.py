#!/usr/bin/env python3
"""
Minimal training test to run on cloud
"""
import torch
import torch.nn as nn
import sys
import os

def test_minimal_training():
    print("🚀 Cloud Training Test - Minimal Setup")
    print("=" * 60)
    
    # Add current directory to path
    sys.path.append(os.getcwd())
    
    # Import our modules
    from tasks import MultiContextMixtureLinear
    from samplers import MultiContextMixtureSampler
    from models import TransformerModel
    
    # Small configuration for quick testing
    n_dims = 10
    batch_size = 4
    n_contexts = 3
    context_length = 8
    predict_length = 1
    
    # Create components
    model = TransformerModel(
        n_dims=n_dims,
        n_positions=100,
        n_embd=128,
        n_layer=4,
        n_head=4
    )
    
    task = MultiContextMixtureLinear(
        n_dims=n_dims,
        batch_size=batch_size,
        n_contexts=n_contexts,
        context_length=context_length,
        predict_length=predict_length
    )
    
    sampler = MultiContextMixtureSampler(
        n_dims=n_dims,
        n_contexts=n_contexts,
        context_length=context_length,
        predict_length=predict_length
    )
    
    # Get sequence structure
    structure = sampler.get_sequence_structure()
    total_points = structure['total_length']
    predict_inds = list(range(structure['predict_position'] + 1, total_points))
    
    print(f"Training setup:")
    print(f"  - Model: {model.name}")
    print(f"  - Sequence length: {total_points}")
    print(f"  - Predicting {len(predict_inds)} positions")
    print(f"  - Batch size: {batch_size}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # Run a few training steps
    for step in range(5):
        # Generate data
        xs = sampler.sample_xs(total_points, batch_size)
        ys = task.evaluate(xs)
        
        # Training step
        optimizer.zero_grad()
        
        # Use multi-context prediction
        output = model(xs, ys, inds=predict_inds, sequence_structure=structure)
        loss = loss_fn(output, ys[:, predict_inds])
        
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}: loss = {loss.item():.6f}")
        
        # Check mixture info on first step
        if step == 0:
            mix_info = task.get_mixture_info()
            print(f"  Components: {mix_info['n_components']}, Contexts: {mix_info['n_contexts']}")
    
    print("✅ Minimal training test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_minimal_training()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)