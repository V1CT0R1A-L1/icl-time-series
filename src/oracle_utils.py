"""
Oracle baseline for group_mixture_linear evaluation.

Oracle knows the STRUCTURE (K clusters of C points, target cluster with T_target + 1)
but NOT the true coefficients. It fits w via least squares from observed (x,y) pairs.
- Context clusters: fit w from (x,y) in that cluster (causal: only prior points).
- Target cluster: fit w from target-context (x,y) pairs, predict query.
- First position of each segment: no data to fit -> predict 0.
- Prior: target cluster uses one of the K context components -> at target's first
  position, predict with uniform mixture over the K fitted context w's.
"""
import torch


def _fit_w_per_batch(seg_x, seg_y, x_query, d, device):
    """Fit w from (seg_x, seg_y) for each batch element; return predictions x_query @ w."""
    B = seg_x.shape[0]
    y_pred = torch.zeros(B, device=device)
    for b in range(B):
        Xb, yb = seg_x[b], seg_y[b]
        n_prev = Xb.shape[0]
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
            y_pred[b] = (x_query[b : b + 1] @ wb).squeeze()
        except Exception:
            y_pred[b] = 0.0
    return y_pred


def _fit_w_tensor(seg_x, seg_y, d, device):
    """Fit w from (seg_x, seg_y) for each batch element; return (B, d) weights."""
    B = seg_x.shape[0]
    ws = torch.zeros(B, d, device=device)
    for b in range(B):
        Xb, yb = seg_x[b], seg_y[b]
        n_prev = Xb.shape[0]
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
            ws[b] = wb.squeeze(-1)
        except Exception:
            pass  # leave zeros
    return ws


def compute_oracle_mse_by_position(
    xs, ys, components, component_assignments, K, C, T_target, scale, output_norm_factor=None
):
    """
    Compute per-position MSE of the least-squares oracle.

    Oracle knows the STRUCTURE and that the TARGET uses one of the K context
    components. It fits w via least squares from observed (x,y) pairs.
    (components, component_assignments, scale are ignored; kept for API compat.)
    """
    B, T, d = xs.shape
    context_length = K * C
    device = xs.device

    # Precompute fitted w for each context cluster (using full cluster data)
    context_ws = torch.zeros(K, B, d, device=device)
    for k in range(K):
        start, end = k * C, (k + 1) * C
        seg_x = xs[:, start:end]
        seg_y = ys[:, start:end]
        context_ws[k] = _fit_w_tensor(seg_x, seg_y, d, device)

    # w_mean[b] = mean over k of context_ws[k,b]; shape (B, d)
    w_mean = context_ws.mean(dim=0)

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
                seg_x = xs[:, start:t]
                seg_y = ys[:, start:t]
                y_pred[:, t] = _fit_w_per_batch(seg_x, seg_y, xs[:, t], d, device)
        else:
            # Target cluster: positions [context_length, T)
            # Prior: target uses one of the K context components
            target_start = context_length
            n_prev = t - target_start
            x_t = xs[:, t]  # (B, d)
            if n_prev == 0:
                # No target data yet: predict with uniform mixture over K context w's
                y_pred[:, t] = (x_t * w_mean).sum(dim=1)
            else:
                seg_x = xs[:, target_start:t]
                seg_y = ys[:, target_start:t]
                y_pred[:, t] = _fit_w_per_batch(seg_x, seg_y, x_t, d, device)

    sq_err = (y_pred - ys) ** 2
    return sq_err.mean(dim=0).cpu().numpy()
