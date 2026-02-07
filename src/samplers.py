import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


class OnTheFlyMixtureLinearSampler(DataSampler):
    """
    Redesigned for proper in-context learning:
    
    Sequence layout per example:
      [Cluster 0: C points from random component] 
      [Cluster 1: C points from random component] 
      ...
      [Cluster K-1: C points from random component]
      [Target Cluster: T points with y-values + 1 point to predict]
    
    The model must:
    1. Learn weight vectors w_0, w_1, ..., w_{K-1} from context clusters
    2. Infer which component is used in the target cluster from first T points
    3. Predict the last target point using that component
    
    Total length = (K × C) + (T + 1)
    """

    def __init__(
        self,
        n_dims,
        n_components=3,
        contexts_per_component=4,
        target_cluster_context_points=2,  # T: number of points with y-values in target cluster
        noise_std=0.0,
        scale=1.0,
        **kwargs,
    ):
        super().__init__(n_dims)
        self.n_components = n_components
        self.contexts_per_component = contexts_per_component
        self.target_cluster_context_points = target_cluster_context_points  # T
        self.noise_std = noise_std
        self.scale = scale
        # Predict only the target query (last position): standard ICL evaluation, full context, clear learning signal.
        self.predict_target_only = kwargs.pop('predict_target_only', True)
        # When False (default): data ordered by clusters (cluster 0, cluster 1, ..., target cluster).
        # When True: shuffle only the K context clusters (0..K*C-1); target cluster (K*C..T-1) stays in order.
        self.shuffle_context_points = kwargs.pop('shuffle_context_points', False)

        # Base sampler for xs (just Gaussian)
        filtered_kwargs = {}
        if "bias" in kwargs:
            filtered_kwargs["bias"] = kwargs["bias"]
        if "scale" in kwargs:
            filtered_kwargs["scale"] = kwargs["scale"]
        self.base_sampler = GaussianSampler(n_dims, **filtered_kwargs)

        # State for the current batch (set in sample_xs)
        self.current_components = None          # (B, K, d, 1)
        self.component_assignments = None       # (B, T)
        self.target_components = None           # (B,) - component used for target cluster
        self.cluster_assignments = None          # (B, K) - which component each cluster uses
        # Total = K context clusters × C points + target cluster (T context + 1 prediction)
        self.total_length = (self.n_components * self.contexts_per_component) + (self.target_cluster_context_points + 1)

    def get_sequence_structure(self):
        """
        Returns sequence structure including keys expected by train.py logging.
        If predict_target_only: predict only the last (target query) position — standard ICL setup, full context.
        Else: predict all positions (autoregressive).
        """
        if self.predict_target_only:
            predict_inds = [self.total_length - 1]
            predict_length = 1
        else:
            predict_inds = list(range(self.total_length))
            predict_length = self.total_length
        return {
            "total_length": self.total_length,
            "predict_inds": predict_inds,
            "context_length": self.contexts_per_component,  # C contexts per component
            "predict_length": predict_length,
        }

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, **kwargs):
        if n_points != self.total_length:
            raise ValueError(
                f"OnTheFlyMixtureLinearSampler expected n_points={self.total_length}, "
                f"got {n_points}"
            )
        # Optional fixed assignments for eval (e.g. context clusters = [0,1], target = 0 or 1)
        fixed_cluster_assignments = kwargs.pop("fixed_cluster_assignments", None)  # (K,) or (B, K)
        fixed_target_component = kwargs.pop("fixed_target_component", None)  # scalar or (B,)

        xs_b = self.base_sampler.sample_xs(
            n_points, b_size, n_dims_truncated=n_dims_truncated, seeds=seeds
        )  # (B, T, d)

        B, T, d = xs_b.shape
        K = self.n_components
        C = self.contexts_per_component
        T_target = self.target_cluster_context_points  # T: points with y-values in target cluster

        # Sample K components per example: w ~ N(0, I) then scaled
        components = torch.randn(B, K, d, 1, device=xs_b.device) * self.scale  # (B,K,d,1)

        if fixed_cluster_assignments is not None and not isinstance(fixed_cluster_assignments, torch.Tensor):
            fixed_cluster_assignments = torch.tensor(fixed_cluster_assignments, dtype=torch.long, device=xs_b.device)
        if fixed_target_component is not None and not isinstance(fixed_target_component, torch.Tensor):
            fixed_target_component = torch.tensor(fixed_target_component, dtype=torch.long, device=xs_b.device)

        # For each example, assign components to clusters (random or fixed)
        component_assignments = torch.zeros(B, T, dtype=torch.long, device=xs_b.device)
        cluster_assignments = torch.zeros(B, K, dtype=torch.long, device=xs_b.device)
        
        for b in range(B):
            if fixed_cluster_assignments is not None:
                cluster_assignments[b] = fixed_cluster_assignments[b] if fixed_cluster_assignments.dim() > 1 else fixed_cluster_assignments
            else:
                perm = torch.randperm(K, device=xs_b.device)
                cluster_assignments[b] = perm
            
            # Fill context clusters: each cluster gets C points from its assigned component
            idx = 0
            for k in range(K):
                cluster_comp = cluster_assignments[b, k].item()
                start = idx
                end = idx + C
                component_assignments[b, start:end] = cluster_comp
                idx = end
            
            if fixed_target_component is not None:
                target_comp = int(fixed_target_component[b].item()) if fixed_target_component.dim() > 0 else int(fixed_target_component.item())
            else:
                target_comp = torch.randint(0, K, (1,), device=xs_b.device).item()
            target_start = K * C
            target_context_end = target_start + T_target
            component_assignments[b, target_start:target_context_end] = target_comp
            component_assignments[b, T - 1] = target_comp  # Prediction point also uses same component
        
        # # Randomize the order of context clusters (but keep target cluster at the end)
        # # This makes the task harder by removing positional cues about which cluster is which
        # context_length = K * C
        
        # # #region agent log
        # try:
        #     with open(log_path, 'a', encoding='utf-8') as f:
        #         log_entry = {
        #             "sessionId": "debug-session",
        #             "runId": "pre-randomization",
        #             "hypothesisId": "cluster-order",
        #             "location": "samplers.py:140",
        #             "message": "Before reordering - component assignments for first example",
        #             "data": {
        #                 "original_component_assignments": component_assignments[0, :context_length].cpu().tolist(),
        #                 "target_cluster_assignments": component_assignments[0, context_length:].cpu().tolist()
        #             },
        #             "timestamp": int(torch.cuda.current_device() * 1000) if torch.cuda.is_available() else 0
        #         }
        #         f.write(json.dumps(log_entry) + '\n')
        # except: pass
        # # #endregion
        
        # for b in range(B):
        #     # Create a random permutation of the K context clusters
        #     cluster_order = torch.randperm(K, device=xs_b.device)
            
        #     # #region agent log
        #     try:
        #         with open(log_path, 'a', encoding='utf-8') as f:
        #             log_entry = {
        #                 "sessionId": "debug-session",
        #                 "runId": "pre-randomization",
        #                 "hypothesisId": "cluster-order",
        #                 "location": "samplers.py:155",
        #                 "message": f"Random cluster order for batch {b}",
        #                 "data": {
        #                     "batch_idx": b,
        #                     "cluster_order": cluster_order.cpu().tolist(),
        #                     "original_cluster_assignments": cluster_assignments[b].cpu().tolist()
        #                 },
        #                 "timestamp": int(torch.cuda.current_device() * 1000) if torch.cuda.is_available() else 0
        #             }
        #             f.write(json.dumps(log_entry) + '\n')
        #     except: pass
        #     # #endregion
            
        #     # Reorder context clusters: create new indices for xs and component_assignments
        #     new_indices = torch.zeros(context_length, dtype=torch.long, device=xs_b.device)
        #     for new_pos, old_cluster_idx in enumerate(cluster_order):
        #         old_start = old_cluster_idx * C
        #         old_end = old_start + C
        #         new_start = new_pos * C
        #         new_end = new_start + C
        #         new_indices[new_start:new_end] = torch.arange(old_start, old_end, device=xs_b.device)
            
        #     # Reorder xs_b for context clusters
        #     xs_b[b, :context_length] = xs_b[b, new_indices]
            
        #     # Reorder component_assignments for context clusters
        #     component_assignments[b, :context_length] = component_assignments[b, new_indices]
            
        #     # #region agent log
        #     try:
        #         with open(log_path, 'a', encoding='utf-8') as f:
        #             log_entry = {
        #                 "sessionId": "debug-session",
        #                 "runId": "post-randomization",
        #                 "hypothesisId": "cluster-order",
        #                 "location": "samplers.py:175",
        #                 "message": f"After reordering - component assignments for batch {b}",
        #                 "data": {
        #                     "batch_idx": b,
        #                     "reordered_component_assignments": component_assignments[b, :context_length].cpu().tolist(),
        #                     "target_cluster_unchanged": component_assignments[b, context_length:].cpu().tolist(),
        #                     "cluster_order_used": cluster_order.cpu().tolist()
        #                 },
        #                 "timestamp": int(torch.cuda.current_device() * 1000) if torch.cuda.is_available() else 0
        #             }
        #             f.write(json.dumps(log_entry) + '\n')
        #     except: pass
        #     # #endregion

        # Optional: shuffle only the K context clusters (positions 0..K*C-1); target cluster (K*C..T-1) stays in order
        # so the model can see the target cluster's (x,y) pairs and query in order.
        if self.shuffle_context_points:
            context_length = K * C
            target_start = K * C
            for b in range(B):
                perm = torch.randperm(context_length, device=xs_b.device)
                new_order = torch.cat([
                    perm,
                    torch.arange(target_start, T, device=xs_b.device, dtype=perm.dtype),
                ])
                xs_b[b] = xs_b[b, new_order]
                component_assignments[b] = component_assignments[b, new_order]

        self.current_components = components
        self.component_assignments = component_assignments
        self.cluster_assignments = cluster_assignments
        self.target_components = component_assignments[:, -1]  # Component for prediction point

        return xs_b

class MultiContextMixtureSampler(DataSampler):
    def __init__(self, n_dims, n_contexts=5, n_components=3, 
                 context_length=20, predict_length=1, **kwargs):
        super().__init__(n_dims)
        self.n_contexts = n_contexts
        self.n_components = n_components
        self.context_length = context_length
        self.predict_length = predict_length

        filtered_kwargs = {}
        if 'bias' in kwargs:
            filtered_kwargs['bias'] = kwargs['bias']
        if 'scale' in kwargs:
            filtered_kwargs['scale'] = kwargs['scale']
        self.base_sampler = GaussianSampler(n_dims, **filtered_kwargs)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        '''
        Structure: 
        [series_1_point_1] [series_1_point_2] ... [series_1_point_k] [SEP]
        [series_2_point_1] [series_2_point_2] ... [series_2_point_k] [SEP]
        ...
        [target_series_point_1] [target_series_point_2] ... [target_series_point_m] [PREDICT] // change to x1 y1 x2 y2
        '''
        structure = self.get_sequence_structure()
        total_length = structure['total_length']
        
        if n_points != total_length:
            raise ValueError(f"Requested {n_points} points but need {total_length} for structure")
        
        xs_b = self.base_sampler.sample_xs(total_length, b_size, n_dims_truncated, seeds)
        
        for b in range(b_size):
            for sep_pos in structure['sep_positions']:
                if sep_pos < total_length:
                    xs_b[b, sep_pos] = torch.zeros(self.n_dims)
            
        return xs_b
    
    def get_sequence_structure(self):
        
        total_length = self.n_contexts * self.context_length + self.n_contexts + 1     

        sep_positions = []
        for i in range(self.n_contexts):
            sep_pos = (i + 1) * self.context_length + i
            sep_positions.append(sep_pos)

        predict_position = total_length - 1
        predict_inds = [predict_position]

        return {
            'n_contexts': self.n_contexts,
            'n_components': self.n_components,
            'context_length': self.context_length,
            'predict_length': self.predict_length,
            'total_length': total_length,
            'sep_positions': sep_positions,
            'predict_position': predict_position,
            'predict_inds': predict_inds
        }
    
class ARWarmupSampler(DataSampler):
    """
    AR(q) time series sampler for warmup experiments.
    Generates actual AR(q) sequences with known coefficients.
    """
    def __init__(self, n_dims, lag=3, base_sampler=None, noise_std=0.2, **kwargs):
        super().__init__(n_dims)
        self.lag = lag
        self.noise_std = noise_std
        # Use GaussianSampler as default base sampler if none provided
        # For AR tasks, always use 1-dimensional base sampler
        if base_sampler is None:
            self.base_sampler = GaussianSampler(1, **kwargs)  # Always 1D for AR
        else:
            self.base_sampler = base_sampler
        self.current_coefficients = None # Gets set in train.py
    
    def generate_bounded_coefficients(self, batch_size):
        """
        Generate AR coefficients that prevent explosive growth.
        Ensures the AR process remains bounded/stationary.
        
        Uses normalization by lag: x_t = (1/d) * sum(x_{t-i} * a_i) where a_i ~ N(0, 1)
        This is more natural and increases stability compared to scaling down variance.
        """
        # Generate coefficients from N(0, 1) and normalize by 1/lag
        coeffs = torch.randn(batch_size, self.lag) / self.lag
        return coeffs
    
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        """
        Generate AR(q) sequences with known coefficients and extract lagged features.
        
        Args:
            n_points: Number of time points to generate
            b_size: Batch size
            n_dims_truncated: Target output dimension (should match model.n_dims)
            seeds: Random seeds for reproducibility
            
        Returns:
            xs: (b_size, n_points, n_dims_truncated) tensor of lagged features
        """        
        # Generate AR sequences using these coefficients
        T = n_points + self.lag
        xs_b = torch.zeros(b_size, n_points, self.n_dims)
        ys_b = torch.zeros(b_size, n_points)  # Store actual noisy next values
        
        if seeds is not None:
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                # Generate initial lag values randomly
                z = torch.zeros(T)
                z[:self.lag] = torch.randn(self.lag, generator=generator)
                
                # Finish the remaining N-q values for AR(q) process
                # Might be slow to parallelize?
                for t in range(self.lag, T):
                    # x_t = (1/d) * sum(x_{t-i} * a_i) + ε_t, where ε_t ~ N(0, noise_std^2)
                    z[t] = sum(self.current_coefficients[i, j] * z[t-1-j] for j in range(self.lag))
                    z[t] += torch.randn(1, generator=generator).item() * self.noise_std
                
                # Create lagged features from this AR sequence
                lagged_features = self._create_lagged_features(z)
                # Pad or crop so that it reaches n_dims dimensions
                if self.lag <= self.n_dims:
                    xs_b[i, :, :self.lag] = lagged_features
                else:
                    xs_b[i] = lagged_features[:, :self.n_dims]
                
                # Store the actual next values (the noisy values, not predictions)
                # xs[i] contains lagged features for predicting z[lag+i]
                # so ys[i] should be the actual value z[lag+i]
                ys_b[i, :] = z[self.lag:self.lag+n_points]
        else:
            for i in range(b_size):
                # Generate initial lag values randomly
                z = torch.zeros(T)
                z[:self.lag] = torch.randn(self.lag)
                
                # Finish the remaining N-q values for AR(q) process
                # Might be slow to parallelize?
                for t in range(self.lag, T):
                    # x_t = (1/d) * sum(x_{t-i} * a_i) + ε_t, where ε_t ~ N(0, noise_std^2)
                    z[t] = sum(self.current_coefficients[i, j] * z[t-1-j] for j in range(self.lag))
                    z[t] += torch.randn(1).item() * self.noise_std
                
                # Create lagged features from this AR sequence
                lagged_features = self._create_lagged_features(z)
                # Pad or truncate to match n_dims
                if self.lag <= self.n_dims:
                    xs_b[i, :, :self.lag] = lagged_features
                else:
                    xs_b[i] = lagged_features[:, :self.n_dims]
                
                # Store the actual next values (the noisy values, not predictions)
                # xs[i] contains lagged features for predicting z[lag+i]
                # so ys[i] should be the actual value z[lag+i]
                ys_b[i, :] = z[self.lag:self.lag+n_points]
        
        if n_dims_truncated is not None:
            xs_b = xs_b[:, :, :n_dims_truncated]
        
        # Store ys_b in the sampler so it can be accessed
        self.current_ys = ys_b
        
        return xs_b
    
    def _create_lagged_features(self, z):
        """
        Create lagged features from a time series.
        
        Args:
            z: (T, 1) time series
            
        Returns:
            lagged: (n_points, lag) lagged features

        e.g. z is (19,) and lagged is (11, 8) if lag = 8
        """
        # Go from (T, 1) -> (T,)
        if z.dim() == 2:
            z = z[:, 0]
        
        # Create lagged indices
        t_idx = torch.arange(self.lag, z.shape[0])  # (n_points,)
        lags = torch.arange(1, self.lag + 1)  # (lag,)
        correct_idx = (t_idx[:, None] - lags[None, :])  # (n_points, lag)

        return z[correct_idx]


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "ar_warmup": ARWarmupSampler,
        "multi_context_mixture": MultiContextMixtureSampler,
        "group_mixture_linear": OnTheFlyMixtureLinearSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            if isinstance(self.scale, (int, float)):
                xs_b = xs_b * self.scale
            else:
                xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
