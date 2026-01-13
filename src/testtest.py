import torch
import math
import numpy as np

# ⚠️ 注意：你需要确保你的 MultiContextMixtureSampler 和 MultiContextMixtureLinear
# 已经从你的项目中正确导入到这里。

from samplers import MultiContextMixtureSampler 
from tasks import MultiContextMixtureLinear
# 假设 Sampler 和 Task 类已经定义在当前环境中

def run_pipeline_test():
    """测试 Sampler 和 Task 是否正确地生成了序列和目标值。"""
    
    # --- 1. 定义测试配置 ---
    N_DIMS = 5
    N_CONTEXTS = 1  # 简化为单上下文
    CONTEXT_LENGTH = 8
    PREDICT_LENGTH = 1
    B_SIZE = 1
    
    # 期望的序列长度 L: CL + SEP + PRED = 8 + 1 + 1 = 10
    EXPECTED_LENGTH = N_CONTEXTS * CONTEXT_LENGTH + N_CONTEXTS + 1
    # 期望的预测点索引 I: L - 1 = 9
    EXPECTED_PREDICT_IND = EXPECTED_LENGTH - 1
    
    print(f"--- 🚀 运行数据管道测试 ---")
    print(f"配置: N_DIMS={N_DIMS}, N_CONTEXTS={N_CONTEXTS}, CL={CONTEXT_LENGTH}")
    print(f"期望序列长度 L={EXPECTED_LENGTH}, 期望预测索引 I={EXPECTED_PREDICT_IND}")
    print("-" * 30)

    # --- 2. 初始化 Sampler 和 Task ---
    try:
        sampler = MultiContextMixtureSampler(
            N_DIMS, n_contexts=N_CONTEXTS, n_components=2, 
            context_length=CONTEXT_LENGTH, predict_length=PREDICT_LENGTH
        )
        task = MultiContextMixtureLinear(
            N_DIMS, B_SIZE, n_contexts=N_CONTEXTS, n_components=2,
            context_length=CONTEXT_LENGTH, predict_length=PREDICT_LENGTH, scale=1.0
        )
    except NameError as e:
        print(f"❌ 错误: 无法找到 Sampler 或 Task 类。请确保类已定义或导入。详细: {e}")
        return

    # --- 3. 验证序列结构 (Sampler) ---
    structure = sampler.get_sequence_structure()
    
    # 校验长度
    if structure['total_length'] != EXPECTED_LENGTH:
        print(f"❌ 结构错误: 实际长度 {structure['total_length']} != 期望长度 {EXPECTED_LENGTH}")
        return
    
    # 校验预测索引
    if structure['predict_inds'][0] != EXPECTED_PREDICT_IND:
        print(f"❌ 结构错误: 实际预测索引 {structure['predict_inds'][0]} != 期望索引 {EXPECTED_PREDICT_IND}")
        return
        
    # 校验 SEP 索引
    sep_pos = structure['sep_positions']
    if sep_pos[0] != CONTEXT_LENGTH:
        print(f"❌ 结构错误: 实际 SEP 索引 {sep_pos[0]} != 期望索引 {CONTEXT_LENGTH}")
        return
        
    print(f"✅ Sampler 结构检查通过 (L={EXPECTED_LENGTH}, I={EXPECTED_PREDICT_IND})")

    # --- 4. 采样 X (Sampler) ---
    xs = sampler.sample_xs(EXPECTED_LENGTH, B_SIZE)
    
    # 检查 X 的维度
    if xs.shape != (B_SIZE, EXPECTED_LENGTH, N_DIMS):
        print(f"❌ X 维度错误: 实际 {xs.shape} != 期望 ({B_SIZE}, {EXPECTED_LENGTH}, {N_DIMS})")
        return
        
    # 检查 SEP Token 是否归零 (索引 8)
    sep_idx = sep_pos[0]
    if torch.any(xs[0, sep_idx] != 0.0):
        print(f"❌ X 值错误: SEP 索引 {sep_idx} 处的 X 值不为零。")
        return
        
    # 检查 Context/Predict X 是否非零 (索引 0 和 9)
    if torch.all(xs[0, 0] == 0.0) or torch.all(xs[0, EXPECTED_PREDICT_IND] == 0.0):
        print("❌ X 值错误: Context 或 Predict 索引处的 X 值为零。")
        return
        
    print(f"✅ Sampler X 值检查通过 (SEP={sep_idx} 为零, Context/Predict 非零)")

    # --- 5. 计算 Y (Task) ---
    ys = task.evaluate(xs)
    
    # 检查 Y 的维度
    if ys.shape != (B_SIZE, EXPECTED_LENGTH):
        print(f"❌ Y 维度错误: 实际 {ys.shape} != 期望 ({B_SIZE}, {EXPECTED_LENGTH})")
        return

    # 检查 SEP Y 值 (索引 8)
    y_sep = ys[0, sep_idx].item()
    if abs(y_sep) > 1e-6:
        print(f"❌ Y 值错误: SEP 索引 {sep_idx} 处的 Y 值不为零 ({y_sep:.4f})。")
        return

    # 检查 Context Y 值 (索引 0)
    y_context = ys[0, 0].item()
    if abs(y_context) < 1e-3:
        print(f"❌ Y 值错误: Context 索引 0 处的 Y 值接近零 ({y_context:.4f})。")
        return

    # 检查 Predict Y 值 (索引 9)
    y_predict = ys[0, EXPECTED_PREDICT_IND].item()
    if abs(y_predict) < 1e-3:
        print(f"❌ Y 值错误: Predict 索引 {EXPECTED_PREDICT_IND} 处的 Y 值接近零 ({y_predict:.4f})。")
        return
        
    # 检查 Y 的尺度 (期望方差 ~1.0)
    y_std = ys[0, :CONTEXT_LENGTH].std().item()
    if not (0.5 < y_std < 2.0):
        print(f"⚠️ 警告: Context Y 的标准差 {y_std:.4f} 不在期望范围 (0.5-2.0)。请检查归一化。")
        
    print(f"✅ Task Y 值检查通过。")
    print(f"   Context Y (0): {y_context:.4f}")
    print(f"   SEP Y ({sep_idx}): {y_sep:.4f} (必须为零)")
    print(f"   Predict Y ({EXPECTED_PREDICT_IND}): {y_predict:.4f} (必须非零)")
    print("-" * 30)
    print("✨ **恭喜！数据管道已通过所有基本检查。**")
    print("如果所有检查都通过，请再次检查 train_step 中的 Loss Masking！")

if __name__ == "__main__":
    # 在这里放置你的 Sampler 和 Task 类定义（如果它们没有被导入）
    # ... [MultiContextMixtureSampler 和 MultiContextMixtureLinear 的代码]
    
    run_pipeline_test()