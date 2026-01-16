import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
# import xgboost as xgb

from base_models import NeuralNetwork, ParallelNetworks


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            # (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, 
                 sep_token_id=-1, predict_token_id=-2):
        super(TransformerModel, self).__init__()
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
        
        # Special token handling
        self.sep_token_id = sep_token_id
        self.predict_token_id = predict_token_id
        self.special_embeddings = nn.Embedding(2, n_embd)  # 2 special tokens: SEP and PREDICT

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

    def _add_special_token_embeddings(self, embeds, sequence_structure):
        """
        Add special token embeddings to the input embeddings.
        
        Args:
            embeds: (batch_size, seq_len, n_embd) input embeddings
            sequence_structure: dict containing special token positions
            
        Returns:
            embeds_with_special: embeddings with special tokens added
        """
        if sequence_structure is None:
            return embeds
            
        batch_size, seq_len, n_embd = embeds.shape
        
        # Get special token positions (convert from xs/ys space to interleaved space)
        sep_positions = [pos * 2 for pos in sequence_structure.get('sep_positions', [])]
        predict_position = sequence_structure.get('predict_position', -1) * 2
        
        # Add SEP token embeddings
        if sep_positions:
            sep_embed = self.special_embeddings(
                torch.tensor([0], device=embeds.device)  # 0 for SEP token
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, n_embd)
            
            for pos in sep_positions:
                if pos < seq_len:
                    embeds[:, pos] = embeds[:, pos] + sep_embed[0, 0]
        
        # Add PREDICT token embedding
        if predict_position >= 0 and predict_position < seq_len:
            predict_embed = self.special_embeddings(
                torch.tensor([1], device=embeds.device)  # 1 for PREDICT token
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, n_embd)
            embeds[:, predict_position] = embeds[:, predict_position] + predict_embed[0, 0]
        
        return embeds

    def _detect_special_tokens_from_data(self, xs):
        """
        Detect special token positions by looking for zero vectors in the input.
        This provides backward compatibility when sequence_structure is not provided.
        
        Args:
            xs: (batch_size, seq_len, n_dims) input sequence
            
        Returns:
            dict with detected special token positions
        """
        batch_size, seq_len, n_dims = xs.shape
        
        # Special tokens are zero vectors in the input
        is_special = (xs == 0).all(dim=-1)  # (batch_size, seq_len)
        
        # For simplicity, use the first batch element to detect structure
        special_positions = torch.where(is_special[0])[0].tolist()
        
        # Try to infer structure: SEP tokens come before PREDICT token
        if len(special_positions) > 0:
            # Assume last special token is PREDICT, others are SEP
            predict_position = special_positions[-1]
            sep_positions = special_positions[:-1]
            
            return {
                'sep_positions': sep_positions,
                'predict_position': predict_position
            }
        
        return None

    def forward(self, xs, ys, inds=None, sequence_structure=None):
        if inds is not None and len(inds) == 0:
            return torch.zeros(xs.shape[0], 0, device=xs.device)

        # Track if inds was originally None (standard autoregressive) vs explicitly provided (selective prediction)
        inds_was_none = (inds is None)
        
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.tensor(inds, device=ys.device)
            if len(inds) > 0:
                valid_inds = [i for i in inds.tolist() if 0 <= i < ys.shape[1]]
                if len(valid_inds) != len(inds):
                    print(
                        f"WARNING: Some indices are out of bounds. Requested: {inds}, "
                        f"Valid: {valid_inds}"
                    )
                    if len(valid_inds) == 0:
                        return torch.zeros(xs.shape[0], 0, device=xs.device)
                    inds = torch.tensor(valid_inds, device=ys.device)

        # Mask labels at prediction indices ONLY if inds was explicitly provided
        # For standard autoregressive (inds=None), don't mask - model sees all y values
        ys_input = ys.clone()
        if not inds_was_none and len(inds) > 0:
            ys_input[:, inds] = 0.0
            # Diagnostic: verify masking worked (only print once)
            if not hasattr(self, '_masking_verified'):
                print(f"MODEL DEBUG: Masked {len(inds)} positions. ys_input at masked pos: {ys_input[0, inds[0]].item():.6f}")
                print(f"MODEL DEBUG: Original ys at masked pos: {ys[0, inds[0]].item():.6f}")
                self._masking_verified = True

        zs = self._combine(xs, ys_input)
        embeds = self._read_in(zs)
        
        # #region agent log
        import json
        import os
        try:
            os.makedirs('.cursor', exist_ok=True)
            log_path = os.path.join('.cursor', 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                target_idx_in_ys = inds[0].item() if len(inds) > 0 else -1
                target_idx_in_zs = target_idx_in_ys * 2 + 1 if target_idx_in_ys >= 0 else -1
                context_y_idx = 1  # First y position (y0)
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "E",
                    "location": "models.py:226",
                    "message": "Input sequence values",
                    "data": {
                        "ys_input_shape": list(ys_input.shape),
                        "ys_input_context": ys_input[0, :target_idx_in_ys].cpu().tolist() if target_idx_in_ys > 0 else [],
                        "ys_input_target": ys_input[0, target_idx_in_ys].item() if target_idx_in_ys >= 0 else 0.0,
                        "zs_target_pos": zs[0, target_idx_in_zs, :5].cpu().tolist() if target_idx_in_zs >= 0 and target_idx_in_zs < zs.shape[1] else [],
                        "zs_context_y_pos": zs[0, context_y_idx, :5].cpu().tolist() if zs.shape[1] > context_y_idx else []
                    },
                    "timestamp": int(torch.cuda.current_device() * 1000) if torch.cuda.is_available() else 0
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        
        # Diagnostic: show what the model sees (only once)
        if not hasattr(self, '_sequence_verified'):
            print(f"\nMODEL SEQUENCE DEBUG:")
            print(f"  xs shape: {xs.shape}, ys_input shape: {ys_input.shape}")
            print(f"  After _combine, zs shape: {zs.shape}")
            print(f"  zs[0, :4, :3] (first 4 positions, first 3 dims): {zs[0, :4, :3].cpu().numpy()}")
            print(f"  For point 0: zs[0, 0] should be x0, zs[0, 1] should be [y0, 0, 0, ...]")
            if len(inds) > 0:
                target_idx_in_ys = inds[0].item()
                target_idx_in_zs = target_idx_in_ys * 2 + 1  # y positions are at odd indices
                print(f"  Target is at ys index {target_idx_in_ys}, which is zs position {target_idx_in_zs}")
                print(f"  zs[0, {target_idx_in_zs}] (target y position, should be masked): {zs[0, target_idx_in_zs, :3].cpu().numpy()}")
            self._sequence_verified = True
        
        if sequence_structure is not None:
            embeds = self._add_special_token_embeddings(embeds, sequence_structure)
         
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        
        # Diagnostic: verify prediction extraction
        if not hasattr(self, '_prediction_verified'):
            print(f"\nMODEL PREDICTION DEBUG:")
            print(f"  output shape: {output.shape}")
            print(f"  prediction shape (before indexing): {prediction.shape}")
            print(f"  prediction[:, 1::2, 0] extracts y positions, shape: {prediction[:, 1::2, 0].shape}")
            print(f"  inds: {inds}")
            print(f"  Final output shape: {prediction[:, 1::2, 0][:, inds].shape}")
            self._prediction_verified = True
        
        # #region agent log
        import json
        import os
        try:
            os.makedirs('.cursor', exist_ok=True)
            log_path = os.path.join('.cursor', 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                y_positions = prediction[:, 1::2, 0]
                final_output = y_positions[:, inds]
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "models.py:259",
                    "message": "Prediction extraction values",
                    "data": {
                        "prediction_shape": list(prediction.shape),
                        "y_positions_shape": list(y_positions.shape),
                        "inds": inds.cpu().tolist() if hasattr(inds, 'cpu') else inds,
                        "final_output_shape": list(final_output.shape),
                        "final_output_sample": final_output[0, :3].cpu().tolist() if final_output.shape[1] >= 3 else final_output[0].cpu().tolist(),
                        "y_positions_sample": y_positions[0, :5].cpu().tolist()
                    },
                    "timestamp": int(torch.cuda.current_device() * 1000) if torch.cuda.is_available() else 0
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        
        # #region agent log
        import os
        try:
            os.makedirs('.cursor', exist_ok=True)
            log_path = os.path.join('.cursor', 'debug.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "models.py:259",
                    "message": "Embedding values at target position",
                    "data": {
                        "embeds_shape": list(embeds.shape),
                        "target_zs_idx": (inds[0].item() * 2 + 1) if len(inds) > 0 else -1,
                        "embeds_at_target_sample": embeds[0, (inds[0].item() * 2 + 1) if len(inds) > 0 else 0, :5].cpu().tolist(),
                        "embeds_context_sample": embeds[0, 1, :5].cpu().tolist() if embeds.shape[1] > 1 else []
                    },
                    "timestamp": int(torch.cuda.current_device() * 1000) if torch.cuda.is_available() else 0
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        
        return prediction[:, 1::2, 0][:, inds]


    def get_special_token_info(self):
        """Return information about special tokens for debugging"""
        return {
            'sep_token_id': self.sep_token_id,
            'predict_token_id': self.predict_token_id,
            'special_embedding_shape': self.special_embeddings.weight.shape
        }
    
class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# class XGBoostModel:
#     def __init__(self):
#         self.name = "xgboost"

#     # inds is a list containing indices where we want the prediction.
#     # prediction made at all indices by default.
#     def __call__(self, xs, ys, inds=None):
#         xs, ys = xs.cpu(), ys.cpu()

#         if inds is None:
#             inds = range(ys.shape[1])
#         else:
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")

#         preds = []

#         # i: loop over num_points
#         # j: loop over bsize
#         for i in tqdm(inds):
#             pred = torch.zeros_like(ys[:, 0])
#             if i > 0:
#                 pred = torch.zeros_like(ys[:, 0])
#                 for j in range(ys.shape[0]):
#                     train_xs, train_ys = xs[j, :i], ys[j, :i]

#                     clf = xgb.XGBRegressor()

#                     clf = clf.fit(train_xs, train_ys)
#                     test_x = xs[j, i : i + 1]
#                     y_pred = clf.predict(test_x)
#                     pred[j] = y_pred[0].item()

#             preds.append(pred)

#         return torch.stack(preds, dim=1)
