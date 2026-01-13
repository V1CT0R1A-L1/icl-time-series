import math

import torch

import numpy as np

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "ar_warmup": ARWarmup,
        "multi_context_mixture_linear": MultiContextMixtureLinear,
        "group_mixture_linear": OnTheFlyMixtureLinear,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError

class MultiContextMixtureLinear(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, 
                 n_contexts=5, n_components=3, context_length=20, 
                 predict_length=1, scale=1.0, **kwargs):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        
        self.n_contexts = n_contexts
        self.n_components = n_components
        self.context_length = context_length
        self.predict_length = predict_length
        self.output_norm_factor = math.sqrt(n_dims)
        self.scale = scale
        
        self.current_components = None
        self.context_assignments = None
        self.target_assignment = None

    def _generate_fresh_components(self, batch_size):
        components = torch.randn(batch_size, self.n_components, self.n_dims, 1)
        return components * self.scale
    
    def _assign_components(self, batch_size):
        context_assignments = torch.randint(0, self.n_components, 
                                          (batch_size, self.n_contexts))
        target_assignments = torch.randint(0, self.n_components, (batch_size,))
        
        return context_assignments, target_assignments
    
    def evaluate(self, xs_b):
        # xs_b: (batch_size, total_points, n_dims)
        # ys_b: (batch_size, total_points)
        
        batch_size, total_points, n_dims = xs_b.shape
        components = self._generate_fresh_components(batch_size)
        context_assignments, target_assignments = self._assign_components(batch_size)
        
        self.current_components = components
        self.context_assignments = context_assignments
        self.target_assignment = target_assignments
        
        points_per_context = self.context_length
        # points_per_target = self.context_length - self.predict_length
        

        total_expected_points = self.n_contexts * self.context_length + self.n_contexts + 1
        
        if total_points != total_expected_points:
            raise ValueError(f"Expected {total_expected_points} points, got {total_points}")

        ys_b = torch.zeros(batch_size, total_points, device=xs_b.device)
        
        for b in range(batch_size):
            point_idx = 0
            
            for ctx_idx in range(self.n_contexts):
                comp_idx = context_assignments[b, ctx_idx]
                w = components[b, comp_idx].to(xs_b.device)
                start, end = point_idx, point_idx + points_per_context
                context_xs = xs_b[b, start:end]
                context_ys = (context_xs @ w)[:, 0] / self.output_norm_factor
                ys_b[b, start:end] = context_ys * self.scale
                point_idx += points_per_context + 1 # for pecial token
                            
            target_comp_idx = target_assignments[b]
            w_target = components[b, target_comp_idx].to(xs_b.device)        

            # start, end = point_idx, point_idx + points_per_target
            # target_xs = xs_b[b, start:end]
            
            # target_ys = (target_xs @ w_target)[:, 0] / self.output_norm_factor
            # ys_b[b, start:end] = target_ys * self.scale

            predict_idx = total_points - 1
            x_to_predict = xs_b[b, predict_idx] 
            y_to_predict = ((x_to_predict @ w_target).squeeze() / self.output_norm_factor) * self.scale            
            ys_b[b, predict_idx] = y_to_predict
                        
        return ys_b
    
    def get_mixture_info(self):
        """Return mixture component information for analysis"""
        return {
            'components': self.current_components,
            'context_assignments': self.context_assignments,
            'target_assignment': self.target_assignment,
            'n_components': self.n_components,
            'n_contexts': self.n_contexts
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class OnTheFlyMixtureLinear(Task):
    """
    Linear-regression-style task for the grouped mixture setting.

    It expects:
      - components: (B, K, n_dims, 1) tensor of weight vectors
      - component_assignments: (B, T) long tensor, each entry in {0..K-1}
    For each position t, y[b,t] = x[b,t]^T w_{components[b, component_assignments[b,t]]}.
    """

    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        components=None,
        component_assignments=None,
        scale=1.0,
        noise_std=0.0,
        **kwargs,
    ):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        assert components is not None, "OnTheFlyMixtureLinear requires components"
        assert component_assignments is not None, "OnTheFlyMixtureLinear requires component_assignments"

        self.components = components
        self.component_assignments = component_assignments.long()
        self.scale = scale
        self.noise_std = noise_std

    def evaluate(self, xs_b):
        """
        xs_b: (B, T, d)
        Returns:
          ys_b: (B, T)
        """
        B, T, d = xs_b.shape
        components = self.components.to(xs_b.device)
        comp_ids = self.component_assignments.to(xs_b.device)

        # Gather the right weight vector per point
        batch_idx = torch.arange(B, device=xs_b.device).unsqueeze(1).expand(B, T)
        # w_for_points: (B, T, d, 1)
        w_for_points = components[batch_idx, comp_ids]

        # Compute y = x^T w
        ys_b = (xs_b.unsqueeze(-2) @ w_for_points).squeeze(-1).squeeze(-1)
        ys_b = ys_b * self.scale

        if self.noise_std > 0:
            ys_b = ys_b + torch.randn_like(ys_b) * self.noise_std

        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class ARWarmup(Task):
    """
    AR(q), all values are scalars.
    xs_b shape: (B, N, q), where each row is last q values of the single series
    Returns y of shape (B, N), where y[b,n] = <w_b[b], xs_b[b,n,:]>.
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, lag=1, scale=1.0, coefficients=None):
        # n_dims is ignored; feature dim is q in this univariate setup
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        assert pool_dict is None and seeds is None
        self.lag, self.scale = lag, scale
        
        # Use provided coefficients if available, otherwise generate random ones
        if coefficients is not None:
            # coefficients should be (batch_size, lag)
            assert coefficients.shape == (batch_size, lag), \
                f"Coefficients shape {coefficients.shape} doesn't match expected ({batch_size}, {lag})"
            self.w_b = coefficients.unsqueeze(-1)  # (batch_size, lag, 1)
        else:
            # Fallback: randomize self.lag values for each batch
            self.w_b = torch.randn(self.b_size, self.lag, 1)

    def evaluate(self, xs_b: torch.Tensor) -> torch.Tensor:
        """
        xs_b: (B, N, n_dims)  -->  y: (B, N)
        y[b,n] = <w_b[b], xs_b[b,n,:lag]> (only uses first lag dimensions)
        """
        if xs_b.ndim != 3 or xs_b.shape[2] < self.lag:
            raise ValueError(f"xs_b must be (B,N,d) where d >= {self.lag}, got {xs_b.shape}")
        
        # Extract only the first 'lag' dimensions for AR computation (make sure we aren't using extra)
        xs_lag = xs_b[:, :, :self.lag]  # (B, N, lag)
        w = self.w_b.to(xs_b.device, xs_b.dtype) # (B, lag, 1)
        y = (xs_lag @ w)[:, :, 0] # (B, N)
        return self.scale * y

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    def get_weights(self):
        return self.w_b


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
