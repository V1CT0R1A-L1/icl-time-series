"""
Oracle baseline for group_mixture_linear evaluation.

Oracle knows the STRUCTURE (K clusters of C points, target cluster with T_target + 1)
but NOT the true coefficients. It fits w via least squares from observed (x,y) pairs.
- Context clusters: fit w from (x,y) in that cluster (causal: only prior points).
- Target cluster: posterior over K context components; predict with weighted average
  of context_ws using p(k|target_data). Posterior updated sequentially as data arrives.
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


def _posterior_and_predict(seg_x, seg_y, context_ws, x_query, sigma, device):
    """
    Compute p(k|data) from target (seg_x, seg_y) and predict y = x_query @ w_eff.
    Likelihood: y_i = x_i @ w_k + eps, eps ~ N(0, sigma^2).
    Returns (B,) predictions.
    """
    B, n_obs, d = seg_x.shape
    K = context_ws.shape[0]
    # log p(data|k) = -0.5/sigma^2 * sum_i (y_i - x_i @ w_k)^2 + const
    log_lik = torch.zeros(B, K, device=device)
    for k in range(K):
        # pred[b,i] = seg_x[b,i] @ context_ws[k,b]
        pred_k = (seg_x * context_ws[k].unsqueeze(1)).sum(dim=2)  # (B, n_obs)
        sq_err = (seg_y - pred_k) ** 2
        log_lik[:, k] = -0.5 * sq_err.sum(dim=1) / (sigma ** 2)
    # Uniform prior: log p(k) = -log(K)
    log_post = log_lik - torch.log(torch.tensor(K, dtype=torch.float32, device=device))
    log_post = log_post - torch.logsumexp(log_post, dim=1, keepdim=True)
    p_k = torch.exp(log_post)  # (B, K)
    # w_eff[b] = sum_k p_k[b,k] * context_ws[k,b]
    w_eff = (p_k.unsqueeze(2) * context_ws.permute(1, 0, 2)).sum(dim=1)  # (B, d)
    y_pred = (x_query * w_eff).sum(dim=1)
    return y_pred


def compute_oracle_mse_by_position(
    xs, ys, components, component_assignments, K, C, T_target, scale, output_norm_factor=None,
    target_noise_std=1.0,
):
    """
    Compute per-position MSE of the least-squares oracle.

    Oracle knows the STRUCTURE and that the TARGET uses one of the K context
    components. Context: fits w via least squares.     Target: posterior over K
    components, predict with p(k|data) * w_k. target_noise_std: noise scale for
    likelihood (default 1.0). (components, component_assignments, scale ignored.)
    """
    B, T, d = xs.shape
    context_length = K * C
    device = xs.device
    sigma = float(target_noise_std) if target_noise_std is not None else 1.0

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
                # Posterior over K components from target data so far; predict with weighted average
                seg_x = xs[:, target_start:t]  # (B, n_prev, d)
                seg_y = ys[:, target_start:t]  # (B, n_prev)
                y_pred[:, t] = _posterior_and_predict(
                    seg_x, seg_y, context_ws, x_t, sigma, device
                )

    sq_err = (y_pred - ys) ** 2
    return sq_err.mean(dim=0).cpu().numpy()
