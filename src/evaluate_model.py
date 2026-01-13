#!/usr/bin/env python3
"""
专门用于评估旧模型的脚本（没有 special_embeddings）
"""
import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from eval import get_model_from_run
from samplers import get_data_sampler
from tasks import get_task_sampler

class LegacyTransformerModel(nn.Module):
    """旧版本的 TransformerModel，没有 special_embeddings"""
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(LegacyTransformerModel, self).__init__()
        from transformers import GPT2Model, GPT2Config
        
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]

def load_legacy_model(run_path):
    """加载旧版本的模型"""
    import yaml
    from munch import Munch
    
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:
        conf = Munch.fromDict(yaml.safe_load(fp))
    
    # 使用旧版本的模型类
    model = LegacyTransformerModel(
        n_dims=conf.model.n_dims,
        n_positions=conf.model.n_positions,
        n_embd=conf.model.n_embd,
        n_layer=conf.model.n_layer,
        n_head=conf.model.n_head,
    )
    
    # 加载状态
    state_path = os.path.join(run_path, "state.pt")
    state = torch.load(state_path, map_location='cpu')
    model.load_state_dict(state["model_state_dict"])
    
    return model, conf

def evaluate_legacy_model(run_path, num_test_batches=5):
    """评估旧版本的模型"""
    print(f"🔍 Evaluating legacy model from: {run_path}")
    
    # 加载模型和配置
    model, conf = load_legacy_model(run_path)
    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size
    
    print(f"Model: {model.name}")
    print(f"Task: {conf.training.task}")
    print(f"n_dims: {n_dims}, batch_size: {batch_size}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Using device: {device}")
    
    # 准备数据采样器
    task_kwargs = {}
    if hasattr(conf.training, 'task_kwargs') and conf.training.task_kwargs:
        if hasattr(conf.training.task_kwargs, '__dict__'):
            task_kwargs = conf.training.task_kwargs.__dict__
        else:
            task_kwargs = conf.training.task_kwargs
    
    data_sampler = get_data_sampler(conf.training.data, n_dims=n_dims, **task_kwargs)
    task_sampler = get_task_sampler(conf.training.task, n_dims, batch_size, **task_kwargs)
    
    # 评估多个批次
    all_losses = []
    
    for batch_idx in range(num_test_batches):
        print(f"\n--- Test Batch {batch_idx + 1}/{num_test_batches} ---")
        
        # 生成数据
        n_points = conf.training.curriculum.points.end
        xs = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)
        task = task_sampler()
        
        # 获取目标值
        if hasattr(data_sampler, 'current_ys'):
            ys = data_sampler.current_ys
        else:
            ys = task.evaluate(xs)
        
        # 模型预测
        with torch.no_grad():
            pred = model(xs.to(device), ys.to(device))
            batch_loss = task.get_metric()(pred.cpu(), ys)
        
        mean_loss = batch_loss.mean().item()
        all_losses.append(mean_loss)
        
        print(f"Batch loss: {mean_loss:.6f}")
        print(f"Data shapes - xs: {xs.shape}, ys: {ys.shape}, pred: {pred.shape}")
    
    # 打印总体结果
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Average loss over {num_test_batches} batches: {sum(all_losses)/len(all_losses):.6f}")
    print(f"Loss range: {min(all_losses):.6f} - {max(all_losses):.6f}")
    
    # 绘制损失图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(all_losses) + 1), all_losses, 'o-', markersize=8)
    plt.xlabel('Test Batch')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Legacy Model Evaluation - {conf.training.task}')
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plot_path = os.path.join(run_path, 'legacy_evaluation_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Plot saved to: {plot_path}")
    
    plt.show()
    
    return all_losses

if __name__ == "__main__":
    run_path = "models/mixture_ar/ac05b139-eaf0-476f-acc2-a89c2d7d967a"
    
    if os.path.exists(run_path):
        losses = evaluate_legacy_model(run_path, num_test_batches=10)
    else:
        print(f"❌ Run path not found: {run_path}")