import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


class OnTheFlyMixtureLinearSampler(DataSampler):
    """
    For each batch element, sample K linear components and C contexts per component,
    plus one target point. The transformer sees a flat sequence of points; grouping
    is implicit in the component_assignments, which are only known to the generator.

    Sequence layout per example (no SEP/PRED tokens):
      [contexts of comp 0] [contexts of comp 1] ... [contexts of comp K-1] [target]
    Total length T = K * C + 1
    """

    def __init__(
        self,
        n_dims,
        n_components=3,
        contexts_per_component=4,
        noise_std=0.0,
        scale=1.0,
        **kwargs,
    ):
        super().__init__(n_dims)
        self.n_components = n_components
        self.contexts_per_component = contexts_per_component
        self.noise_std = noise_std
        self.scale = scale

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
        self.target_components = None           # (B,)
        self.total_length = self.n_components * self.contexts_per_component + 1

    def get_sequence_structure(self):
        """
        Returns sequence structure including keys expected by train.py logging.
        """
        predict_position = self.total_length - 1
        predict_inds = [predict_position]
        return {
            "total_length": self.total_length,
            "predict_inds": predict_inds,
            "context_length": self.contexts_per_component,  # C contexts per component
            "predict_length": 1,  # We predict one target point
        }

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if n_points != self.total_length:
            raise ValueError(
                f"OnTheFlyMixtureLinearSampler expected n_points={self.total_length}, "
                f"got {n_points}"
            )

        xs_b = self.base_sampler.sample_xs(
            n_points, b_size, n_dims_truncated=n_dims_truncated, seeds=seeds
        )  # (B, T, d)

        B, T, d = xs_b.shape
        K = self.n_components
        C = self.contexts_per_component

        # Sample K components per example: w ~ N(0, I) then scaled
        components = torch.randn(B, K, d, 1, device=xs_b.device) * self.scale  # (B,K,d,1)

        # For each example, assign contexts deterministically and target randomly
        component_assignments = torch.zeros(B, T, dtype=torch.long, device=xs_b.device)
        for b in range(B):
            # Context blocks: 0..C-1 -> comp 0, C..2C-1 -> comp 1, ...
            idx = 0
            for k in range(K):
                start = idx
                end = idx + C
                component_assignments[b, start:end] = k
                idx = end
            # Target index is last, choose a random component
            target_comp = torch.randint(0, K, (1,), device=xs_b.device).item()
            component_assignments[b, T - 1] = target_comp

        self.current_components = components
        self.component_assignments = component_assignments
        self.target_components = component_assignments[:, -1]

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
