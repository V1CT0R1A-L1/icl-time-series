# test_data_independence.py
import torch
import sys
sys.path.append('src')

def test_component_independence():
    """测试不同批次是否使用独立的组件权重"""
    
    from samplers import OnTheFlyMixtureLinearSampler
    
    sampler = OnTheFlyMixtureLinearSampler(
        n_dims=5,
        n_components=10,
        contexts_per_component=2,
        target_cluster_context_points=3,
        scale=0.1
    )
    
    # 生成两个不同的批次
    xs1 = sampler.sample_xs(24, batch_size=2)
    comp1 = sampler.current_components.clone()
    
    xs2 = sampler.sample_xs(24, batch_size=2) 
    comp2 = sampler.current_components.clone()
    
    print("=== 组件独立性测试 ===")
    print(f"批次1组件形状: {comp1.shape}")
    print(f"批次2组件形状: {comp2.shape}")
    
    # 检查批次0和批次1是否相同
    batch_same = torch.allclose(comp1[0], comp1[1])
    print(f"批次内组件相同?: {batch_same}")
    
    # 检查不同批次是否相同
    across_same = torch.allclose(comp1[0], comp2[0])
    print(f"跨批次组件相同?: {across_same}")
    
    # 检查目标权重是否来自现有组件
    target_comp_idx = sampler.target_components[0].item()
    print(f"\n批次0的目标使用组件: {target_comp_idx}")
    print(f"上下文簇中组件{target_comp_idx}的权重: {comp1[0, target_comp_idx, :3, 0]}")
    
    # 如果目标组件权重与上下文簇的完全一致，就有问题！
    
    return not (batch_same or across_same)

if __name__ == "__main__":
    independent = test_component_independence()
    if independent:
        print("\n✓ 组件是独立的")
    else:
        print("\n✗ 问题: 组件可能共享了！")