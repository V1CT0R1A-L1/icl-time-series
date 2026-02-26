"""
Oracle baseline for group_mixture_linear evaluation.

Bayesian oracle: knows the structure (K, C, T_target) and the K components from context.
- Context positions: knows component from structure -> perfect prediction.
- Target cluster: infers component from observed (x,y) pairs via least-squares fit.
  At first target position (no target context yet): predicts uniform mixture over K.
"""
import torch


def compute_oracle_mse_by_position(
    xs, ys, components, component_assignments, K, C, T_target, scale, output_norm_factor=None
):
    """
    Compute per-position MSE of the Bayesian oracle for group_mixture_linear.

    Args:
        xs: (B, T, d)
        ys: (B, T)
        components: (B, K, d, 1)
        component_assignments: (B, T)
        K, C, T_target: structure params
        scale: output scale
        output_norm_factor: if used in task (e.g. sqrt(n_dims)), else 1.0

    Returns:
        mse_by_pos: (T,) mean squared error per position
    """
    if output_norm_factor is None:
        output_norm_factor = 1.0

    B, T, d = xs.shape
    context_length = K * C
    device = xs.device

    # All predictions for context: y = (x @ w_c) * scale / norm
    batch_idx = torch.arange(B, device=device)
    y_pred = torch.zeros_like(ys)

    # Context: oracle knows component
    comp_at_pos = component_assignments  # (B, T)
    w_at_pos = components[batch_idx.unsqueeze(1), comp_at_pos]  # (B, T, d, 1)
    y_pred_all = (xs.unsqueeze(-2) @ w_at_pos).squeeze(-1).squeeze(-1) * scale / output_norm_factor
    y_pred[:, :context_length] = y_pred_all[:, :context_length]

    # Target cluster: iterate over positions (causal: each position uses only prior target context)
    for t in range(context_length, T):
        if t == context_length:
            # Uniform mixture over K: (B, 1, d) @ (B, d, K) -> (B, 1, K)
            comp_flat = components[:, :, :, 0].transpose(1, 2)  # (B, d, K)
            preds_c = (xs[:, t:t + 1] @ comp_flat).squeeze(1)  # (B, K)
            y_pred[:, t] = preds_c.mean(dim=1) * scale / output_norm_factor
        else:
            ctx_x = xs[:, context_length:t]  # (B, n_ctx, d)
            ctx_y = ys[:, context_length:t]  # (B, n_ctx)
            x_t = xs[:, t]  # (B, d)
            for b in range(B):
                best_err = float("inf")
                best_c = 0
                for c in range(K):
                    w = components[b, c]  # (d, 1)
                    y_hat = (ctx_x[b] @ w).squeeze() * scale / output_norm_factor
                    err = ((y_hat - ctx_y[b]) ** 2).sum().item()
                    if err < best_err:
                        best_err = err
                        best_c = c
                y_pred[b, t] = (x_t[b] @ components[b, best_c]).squeeze() * scale / output_norm_factor

    sq_err = (y_pred - ys) ** 2
    return sq_err.mean(dim=0).cpu().numpy()
