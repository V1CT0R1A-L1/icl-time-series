"""
Oracle baseline for group_mixture_linear evaluation.

Oracle knows the STRUCTURE (K clusters of C points, target cluster with T_target + 1)
but NOT the true coefficients. It fits w via least squares from observed (x,y) pairs.
- Context clusters: fit w from (x,y) in that cluster (causal: only prior points).
- Target cluster: fit w from target-context (x,y) pairs, predict query.
- First position of each segment: no data to fit -> predict 0.
"""
import torch


def compute_oracle_mse_by_position(
    xs, ys, components, component_assignments, K, C, T_target, scale, output_norm_factor=None
):
    """
    Compute per-position MSE of the least-squares oracle.

    Oracle knows the STRUCTURE but NOT the true coefficients. It fits w via
    least squares from observed (x,y) pairs in each segment, then predicts.
    (components, component_assignments, scale are ignored; kept for API compat.)
    """
    B, T, d = xs.shape
    context_length = K * C
    device = xs.device

    y_pred = torch.zeros_like(ys)

    # Iterate over positions; fit w from prior (x,y) in segment via least squares
    for t in range(T):
        if t < context_length:
            # Context: which cluster? cluster k has positions [k*C, (k+1)*C)
            k = t // C
            start = k * C
            # Causal: use positions [start, t) from this cluster
            n_prev = t - start
            if n_prev == 0:
                y_pred[:, t] = 0.0
            else:
                seg_x = xs[:, start:t]  # (B, n_prev, d)
                seg_y = ys[:, start:t]  # (B, n_prev)
                x_t = xs[:, t]  # (B, d)
                for b in range(B):
                    Xb = seg_x[b]  # (n_prev, d)
                    yb = seg_y[b]  # (n_prev,)
                    try:
                        if n_prev >= d:
                            wb, _, _, _ = torch.linalg.lstsq(
                                Xb.T @ Xb + 1e-6 * torch.eye(d, device=device),
                                Xb.T @ yb.unsqueeze(1),
                                rcond=None,
                            )
                        else:
                            wb, _, _, _ = torch.linalg.lstsq(
                                Xb, yb.unsqueeze(1), rcond=None
                            )
                        y_pred[b, t] = (x_t[b : b + 1] @ wb).squeeze()
                    except Exception:
                        y_pred[b, t] = 0.0
        else:
            # Target cluster: positions [context_length, T)
            target_start = context_length
            n_prev = t - target_start
            if n_prev == 0:
                y_pred[:, t] = 0.0
            else:
                seg_x = xs[:, target_start:t]  # (B, n_prev, d)
                seg_y = ys[:, target_start:t]  # (B, n_prev)
                x_t = xs[:, t]  # (B, d)
                for b in range(B):
                    Xb = seg_x[b]  # (n_prev, d)
                    yb = seg_y[b]  # (n_prev,)
                    try:
                        if n_prev >= d:
                            wb, _, _, _ = torch.linalg.lstsq(
                                Xb.T @ Xb + 1e-6 * torch.eye(d, device=device),
                                Xb.T @ yb.unsqueeze(1),
                                rcond=None,
                            )
                        else:
                            wb, _, _, _ = torch.linalg.lstsq(
                                Xb, yb.unsqueeze(1), rcond=None
                            )
                        y_pred[b, t] = (x_t[b : b + 1] @ wb).squeeze()
                    except Exception:
                        y_pred[b, t] = 0.0

    sq_err = (y_pred - ys) ** 2
    return sq_err.mean(dim=0).cpu().numpy()
