import torch
import sys
sys.path.append('src')

def emergency_check():
    print("=== 紧急诊断 ===")
    
    # 1. 测试scale的影响
    d = 20
    scale = 1.0  # 你的配置
    
    # 模拟组件权重
    w = torch.randn(d, 1) * scale
    print(f"组件权重w (scale={scale}):")
    print(f"  范数: {torch.norm(w):.4f}")
    print(f"  范围: [{w.min():.4f}, {w.max():.4f}]")
    
    # 模拟x值
    x = torch.randn(1, d)
    print(f"\nx值:")
    print(f"  范数: {torch.norm(x):.4f}")
    print(f"  范围: [{x.min():.4f}, {x.max():.4f}]")
    
    # 计算y = x·w
    y = (x @ w).item()
    print(f"\ny = x·w = {y:.4f}")
    print(f"|y| 可能很大因为 |x|≈{torch.norm(x):.2f}, |w|≈{torch.norm(w):.2f}")
    
    # 2. 计算期望的y范围
    # 对于标准正态x和w: E[|x|] ≈ sqrt(d), E[|w|] ≈ scale*sqrt(d)
    expected_y_std = scale * d  # 近似
    print(f"\n期望的y标准差: ~{expected_y_std:.2f}")
    
    # 3. 建议的scale
    recommended_scale = 1.0 / (d ** 0.5)  # 约0.22
    print(f"建议scale: {recommended_scale:.3f} (1/sqrt(d))")
    
    return recommended_scale

if __name__ == "__main__":
    emergency_check()