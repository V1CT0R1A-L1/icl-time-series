"""
Baselines for group_mixture_linear evaluation.

- **Ground truth (known mixture)**: true w_k per position — ``compute_ground_truth_mixture_mse_by_position``.
- **Bayesian mixture**: causal LS per context cluster; on the target, posterior over
  K context components from target (x,y) — ``compute_bayesian_mixture_mse_by_position``.
- **Pure LS**: same context as above; on the target, a single causal LS model fit only
  to target-segment (x,y) (ignores mixture over K) — ``compute_pure_ls_target_mse_by_position``.
- **Hybrid**: on the target only, ``alpha`` * Bayesian + (1-alpha) * pure-LS target —
  ``compute_hybrid_bayesian_ls_mse_by_position``.

Use ``compute_all_group_mixture_baselines_mse_by_position`` for a dict of all curves (extensible).
"""
from collections import OrderedDict

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


def _predictions_context_clusters_causal(xs, ys, K, C, d, T, context_length, device):
    """Causal LS within each context cluster; first point in each cluster predicts 0."""
    y_pred = torch.zeros(xs.shape[0], T, device=device)
    for t in range(context_length):
        k = t // C
        start = k * C
        n_prev = t - start
        if n_prev == 0:
            y_pred[:, t] = 0.0
        else:
            seg_x = xs[:, start:t]
            seg_y = ys[:, start:t]
            y_pred[:, t] = _fit_w_per_batch(seg_x, seg_y, xs[:, t], d, device)
    return y_pred


def _precompute_context_ws(xs, ys, K, C, d, device):
    """Fitted w from full context segment per cluster: (K, B, d)."""
    context_ws = torch.zeros(K, xs.shape[0], d, device=device)
    for k in range(K):
        start, end = k * C, (k + 1) * C
        context_ws[k] = _fit_w_tensor(xs[:, start:end], ys[:, start:end], d, device)
    return context_ws


def _predictions_group_mixture_target_branch(
    xs, ys, K, C, d, T, context_length, device, sigma, context_ws, y_pred, target_mode,
):
    """
    Fill target positions [context_length, T) in y_pred.
    target_mode: \"bayesian\" | \"pure_ls_target\"
    """
    w_mean = context_ws.mean(dim=0)
    target_start = context_length
    for t in range(context_length, T):
        n_prev = t - target_start
        x_t = xs[:, t]
        if n_prev == 0:
            if target_mode == "bayesian":
                y_pred[:, t] = (x_t * w_mean).sum(dim=1)
            else:
                y_pred[:, t] = 0.0
        else:
            seg_x = xs[:, target_start:t]
            seg_y = ys[:, target_start:t]
            if target_mode == "bayesian":
                y_pred[:, t] = _posterior_and_predict(
                    seg_x, seg_y, context_ws, x_t, sigma, device
                )
            else:
                y_pred[:, t] = _fit_w_per_batch(seg_x, seg_y, x_t, d, device)
    return y_pred


def _predictions_bayesian_mixture_inner(xs, ys, K, C, T_target, target_noise_std):
    """Full-sequence predictions: causal LS per context cluster + Bayesian target."""
    B, T, d = xs.shape
    context_length = K * C
    device = xs.device
    sigma = float(target_noise_std) if target_noise_std is not None else 1.0
    context_ws = _precompute_context_ws(xs, ys, K, C, d, device)
    y_pred = _predictions_context_clusters_causal(xs, ys, K, C, d, T, context_length, device)
    return _predictions_group_mixture_target_branch(
        xs, ys, K, C, d, T, context_length, device, sigma, context_ws, y_pred, "bayesian",
    )


def _predictions_pure_ls_target_inner(xs, ys, K, C, T_target, target_noise_std):
    """Same context as Bayesian; target = single causal LS on target segment only."""
    B, T, d = xs.shape
    context_length = K * C
    device = xs.device
    sigma = float(target_noise_std) if target_noise_std is not None else 1.0
    context_ws = _precompute_context_ws(xs, ys, K, C, d, device)
    y_pred = _predictions_context_clusters_causal(xs, ys, K, C, d, T, context_length, device)
    return _predictions_group_mixture_target_branch(
        xs, ys, K, C, d, T, context_length, device, sigma, context_ws, y_pred, "pure_ls_target",
    )


def _mse_from_predictions(y_pred, ys):
    return ((y_pred - ys) ** 2).mean(dim=0).detach().cpu().numpy()


def compute_ground_truth_mixture_mse_by_position(xs, ys, components, component_assignments, scale):
    """
    Per-position MSE when w_k is known at every position: uses ``components[b, k]`` with
    k = ``component_assignments[b, t]`` (same mean as the generative task).

    Squared error vs observed ``ys`` reflects observation noise where ``noise_std > 0``.
    """
    B, T, d = xs.shape
    device = xs.device
    components = components.to(device)
    comp_ids = component_assignments.to(device).long()
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
    w_for_points = components[batch_idx, comp_ids]
    y_pred = (xs.unsqueeze(-2) @ w_for_points).squeeze(-1).squeeze(-1) * scale
    sq_err = (y_pred - ys) ** 2
    return sq_err.mean(dim=0).cpu().numpy()


def compute_bayesian_mixture_mse_by_position(
    xs, ys, components, component_assignments, K, C, T_target, scale, output_norm_factor=None,
    target_noise_std=1.0,
):
    """
    Per-position MSE for the Bayesian mixture baseline (structure known, w inferred).

    Context: least-squares fit (causal). Target: posterior over K components from
    target data; ``components`` / ``component_assignments`` are unused (kept for API
    symmetry with the ground-truth baseline). ``target_noise_std`` scales the Gaussian likelihood.
    """
    y_pred = _predictions_bayesian_mixture_inner(xs, ys, K, C, T_target, target_noise_std)
    return _mse_from_predictions(y_pred, ys)


def compute_pure_ls_target_mse_by_position(
    xs, ys, components, component_assignments, K, C, T_target, scale, output_norm_factor=None,
    target_noise_std=1.0,
):
    """
    Causal LS within each context cluster (same as Bayesian context). On the target
    segment, a single linear model fit by causal LS to target (x,y) only — no mixture
    over the K components.
    """
    y_pred = _predictions_pure_ls_target_inner(xs, ys, K, C, T_target, target_noise_std)
    return _mse_from_predictions(y_pred, ys)


def compute_hybrid_bayesian_ls_mse_by_position(
    xs, ys, components, component_assignments, K, C, T_target, scale, output_norm_factor=None,
    target_noise_std=1.0,
    hybrid_alpha=0.5,
):
    """
    Context: same as Bayesian / pure LS. Target positions: ``hybrid_alpha`` * Bayesian
    prediction + (1 - ``hybrid_alpha``) * pure-LS-on-target-only prediction (default 0.5 each).
    """
    y_b = _predictions_bayesian_mixture_inner(xs, ys, K, C, T_target, target_noise_std)
    y_p = _predictions_pure_ls_target_inner(xs, ys, K, C, T_target, target_noise_std)
    context_length = K * C
    y_h = y_b.clone()
    a = float(hybrid_alpha)
    y_h[:, context_length:] = (
        a * y_b[:, context_length:] + (1.0 - a) * y_p[:, context_length:]
    )
    return _mse_from_predictions(y_h, ys)


def compute_all_group_mixture_baselines_mse_by_position(
    xs, ys, components, component_assignments, K, C, T_target, scale,
    target_noise_std=1.0,
    hybrid_alpha=0.5,
):
    """
    All built-in baselines as (name -> (T,) numpy MSE per position).

    Order is stable for plotting: ground truth, Bayesian, pure LS target, hybrid.
    Extend this dict when adding new methods.
    """
    out = OrderedDict()
    out["ground_truth"] = compute_ground_truth_mixture_mse_by_position(
        xs, ys, components, component_assignments, scale
    )
    out["bayesian_mixture"] = compute_bayesian_mixture_mse_by_position(
        xs, ys, components, component_assignments, K, C, T_target, scale,
        target_noise_std=target_noise_std,
    )
    out["pure_ls_target"] = compute_pure_ls_target_mse_by_position(
        xs, ys, components, component_assignments, K, C, T_target, scale,
        target_noise_std=target_noise_std,
    )
    out["hybrid_bayesian_ls"] = compute_hybrid_bayesian_ls_mse_by_position(
        xs, ys, components, component_assignments, K, C, T_target, scale,
        target_noise_std=target_noise_std,
        hybrid_alpha=hybrid_alpha,
    )
    return out


# Backward-compatible names
compute_oracle_mse_by_position = compute_bayesian_mixture_mse_by_position
compute_true_mixture_oracle_mse_by_position = compute_ground_truth_mixture_mse_by_position
