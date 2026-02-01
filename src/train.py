import os
import argparse
from random import randint
import uuid

from tqdm import tqdm
import torch
import yaml
import numpy as np

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from models import build_model

import wandb


torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, predict_inds=None, sequence_structure=None, loss_inds=None):
    """loss_inds: optional list of indices into predict_inds to use for loss (e.g. exclude first-of-segment)."""
    optimizer.zero_grad()
    
    if predict_inds is None or len(predict_inds) == 0:
        output = model(xs, ys)
        loss = loss_func(output, ys)
    else:
        if sequence_structure is not None:
            output = model(xs, ys, inds=predict_inds, sequence_structure=sequence_structure)
        else:
            output = model(xs, ys, inds=predict_inds)
        if loss_inds is not None and len(loss_inds) > 0:
            # Restrict loss to positions with in-context signal (e.g. exclude first of each segment).
            pred_sub = output[:, loss_inds]
            tgt_sub = ys[:, [predict_inds[j] for j in loss_inds]]
            loss = loss_func(pred_sub, tgt_sub)
        else:
            loss = loss_func(output, ys[:, predict_inds])
    
    # #region agent log
    import json
    import os
    try:
        os.makedirs('.cursor', exist_ok=True)
        log_path = os.path.join('.cursor', 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "C",
                "location": "train.py:33",
                "message": "Loss computation values",
                "data": {
                    "output_shape": list(output.shape),
                    "output_sample": output[0, :3].cpu().tolist() if output.shape[1] >= 3 else output[0].cpu().tolist(),
                    "target_shape": list(ys[:, predict_inds].shape) if predict_inds is not None else list(ys.shape),
                    "target_sample": ys[0, predict_inds[:3]].cpu().tolist() if predict_inds is not None and len(predict_inds) >= 3 else (ys[0, :3].cpu().tolist() if predict_inds is None else ys[0, predict_inds].cpu().tolist()),
                    "loss_value": loss.item()
                },
                "timestamp": int(torch.cuda.current_device() * 1000) if torch.cuda.is_available() else 0
            }
            f.write(json.dumps(log_entry) + '\n')
    except: pass
    # #endregion
    
    loss.backward()
    
    # #region agent log
    import os
    try:
        log_path = os.path.join('.cursor', 'debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            grad_norms = {}
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    grad_norms[name] = param_norm.item()
            total_norm = total_norm ** (1. / 2)
            
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "D",
                "location": "train.py:35",
                "message": "Gradient norms after backward",
                "data": {
                    "total_grad_norm": total_norm,
                    "grad_norms_sample": dict(list(grad_norms.items())[:5]),
                    "num_params_with_grad": len([p for p in model.parameters() if p.grad is not None]),
                    "num_params_total": len(list(model.parameters()))
                },
                "timestamp": int(torch.cuda.current_device() * 1000) if torch.cuda.is_available() else 0
            }
            f.write(json.dumps(log_entry) + '\n')
    except: pass
    # #endregion
    
    optimizer.step()
    return loss.detach().item(), output.detach()

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    resume_if_exists = getattr(args.training, "resume", True)
    if resume_if_exists and os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()
        print(f"Resumed from step {starting_step} (state.pt)")
    elif not resume_if_exists and os.path.exists(state_path):
        print("resume: false — ignoring existing state.pt, starting from step 0")

    n_dims = model.n_dims
    print("n_dims: ", n_dims)
    bsize = args.training.batch_size
            
    # Convert Args object to dict for task_kwargs
    task_kwargs = {}
    if hasattr(args.training, 'task_kwargs') and args.training.task_kwargs:
        if hasattr(args.training.task_kwargs, '__dict__'):
            task_kwargs = args.training.task_kwargs.__dict__
        else:
            task_kwargs = args.training.task_kwargs
    
    print("task_kwargs: ", task_kwargs)
    # Configure data sampler based on task type
    if args.training.task == "ar_warmup":
        # Use curriculum-controlled lag if available, otherwise fall back to task_kwargs
        lag_value = curriculum.lag
        assert lag_value is not None, "lag must be provided"
        assert isinstance(lag_value, int), "lag must be an integer"
        # Allow noise_std to be configured via task_kwargs
        noise_std = task_kwargs.get('noise_std', 0.2)
        # For AR tasks, use model's n_dims to ensure consistent input dimension
        data_sampler = get_data_sampler("ar_warmup", n_dims=n_dims, lag=lag_value, noise_std=noise_std)
    elif args.training.task == "multi_context_mixture_linear":
        # For multi-context tasks, use the multi_context_mixture sampler
        data_sampler = get_data_sampler("multi_context_mixture", n_dims=n_dims, **task_kwargs)
    elif args.training.task == "group_mixture_linear":
        # On-the-fly grouped mixture linear sampler
        data_sampler = get_data_sampler("group_mixture_linear", n_dims=n_dims, **task_kwargs)
    else:
        data_sampler = get_data_sampler(args.training.data, n_dims=args.training.curriculum.dims.start)

    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **task_kwargs,
    )
    # When resuming, optionally run extra steps: end_step = starting_step + resume_extra_steps
    resume_extra = getattr(args.training, 'resume_extra_steps', None)
    if starting_step > 0 and resume_extra is not None and resume_extra > 0:
        end_step = starting_step + resume_extra
        print(f"Resume mode: training for {resume_extra} more steps (until step {end_step})")
    else:
        end_step = args.training.train_steps
    pbar = tqdm(range(starting_step, end_step))

    num_training_examples = args.training.num_training_examples

    sequence_structure = None
    predict_inds = None
    if hasattr(data_sampler, 'get_sequence_structure'):
        sequence_structure = data_sampler.get_sequence_structure()
        predict_inds = sequence_structure.get('predict_inds', [])
        print(f"Multi-context mode: predicting indices {predict_inds}")
        
        if not predict_inds and 'predict_position' in sequence_structure:
            predict_start = sequence_structure['predict_position'] + 1
            predict_inds = list(range(predict_start, sequence_structure['total_length']))
            print(f"Calculated predict indices: {predict_inds}")
    
    for i in pbar:
        # LR warmup so loss descends smoothly in noiseless ICL
        warmup = getattr(args.training, 'lr_warmup_steps', 0)
        base_lr = args.training.learning_rate
        if warmup > 0 and i < warmup:
            for g in optimizer.param_groups:
                g['lr'] = base_lr * (i + 1) / warmup
        else:
            for g in optimizer.param_groups:
                g['lr'] = base_lr

        data_sampler_args = {}
        task_sampler_args = {}

        if "multi_context" in args.training.task and curriculum.lag is not None:
            # get curriculum train in later
            pass

        if args.training.task == "ar_warmup" and curriculum.lag is not None:
            current_lag = curriculum.lag
            if current_lag != getattr(data_sampler, 'lag', None):
                # Preserve noise_std when recreating sampler
                noise_std = getattr(data_sampler, 'noise_std', 0.2)
                data_sampler = get_data_sampler("ar_warmup", n_dims=n_dims, lag=current_lag, noise_std=noise_std)
            # Update task_kwargs with current lag for task sampler
            task_sampler_args["lag"] = current_lag

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        # Generate bounded coefficients, shouldn't blow up
        if args.training.task == "ar_warmup":
            if data_sampler.current_coefficients is None or not hasattr(data_sampler, 'current_coefficients'):
                data_sampler.current_coefficients = data_sampler.generate_bounded_coefficients(bsize)
        
        if sequence_structure is not None:
            # Use the fixed structure for multi-context tasks
            n_points_to_sample = sequence_structure['total_length']
        else:
            # Use curriculum for other tasks
            n_points_to_sample = curriculum.n_points

        max_allowed_length = model.n_positions // 2 
        if n_points_to_sample > max_allowed_length:
            print(f"SKIPPING: Requested {n_points_to_sample} points but model limit is {max_allowed_length}")
            curriculum.update()
            continue    

        xs = data_sampler.sample_xs(
            n_points_to_sample,  # Use the calculated value
            bsize,
            n_dims_truncated=curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        if xs.shape[1] > max_allowed_length:
            print(f"SKIPPING: Generated {xs.shape[1]} points but model limit is {max_allowed_length}")
            curriculum.update()
            continue
        

        # For AR tasks, pass coefficients from data sampler to task sampler
        if args.training.task == "ar_warmup" and hasattr(data_sampler, 'current_coefficients') and data_sampler.current_coefficients is not None:
            task_sampler_args["coefficients"] = data_sampler.current_coefficients

        # Get targets / build task
        if args.training.task == "group_mixture_linear":
            assert hasattr(data_sampler, "current_components"), "Sampler must set current_components"
            assert hasattr(data_sampler, "component_assignments"), "Sampler must set component_assignments"

            task = task_sampler(
                components=data_sampler.current_components,
                component_assignments=data_sampler.component_assignments,
                **task_sampler_args,
            )
            ys = task.evaluate(xs)
            
            # Diagnostic prints for first batch
            if i == 0:
                print("\n" + "="*60)
                print("DIAGNOSTIC: First batch data structure (Redesigned ICL)")
                print("="*60)
                print(f"xs shape: {xs.shape}")
                print(f"ys shape: {ys.shape}")
                K = data_sampler.n_components
                C = data_sampler.contexts_per_component
                T_target = data_sampler.target_cluster_context_points
                total_context = K * C
                target_start = total_context
                target_context_end = target_start + T_target
                predict_idx = predict_inds[0] if predict_inds is not None and len(predict_inds) > 0 else -1
                
                print(f"\nStructure: {K} context clusters × {C} points + target cluster ({T_target} context + 1 prediction)")
                print(f"Component assignments (first example): {data_sampler.component_assignments[0].cpu().tolist()}")
                if hasattr(data_sampler, 'cluster_assignments'):
                    print(f"Cluster assignments (which component each cluster uses): {data_sampler.cluster_assignments[0].cpu().tolist()}")
                print(f"Target component (first example): {data_sampler.target_components[0].item()}")
                
                # Show context clusters with component weights
                print(f"\nContext clusters (first example):")
                components = data_sampler.current_components[0]  # (K, d, 1)
                for cluster_idx in range(K):
                    cluster_start = cluster_idx * C
                    cluster_end = cluster_start + C
                    cluster_comp = data_sampler.component_assignments[0, cluster_start].item()
                    cluster_ys = ys[0, cluster_start:cluster_end].cpu().numpy()
                    cluster_xs = xs[0, cluster_start:cluster_end, :5].cpu().numpy()  # First 5 dims
                    cluster_w = components[cluster_comp, :5, 0].cpu().numpy()  # First 5 dims of weight
                    print(f"  Cluster {cluster_idx}: uses component {cluster_comp}")
                    print(f"    Weight vector (first 5 dims): {cluster_w}")
                    print(f"    xs (first 5 dims): {cluster_xs}")
                    print(f"    ys: {cluster_ys}")
                    # Verify: y should be approximately x^T w * scale
                    # Note: components are already scaled, and task.evaluate() applies scale again
                    manual_ys = (xs[0, cluster_start:cluster_end] @ components[cluster_comp]).squeeze().cpu().numpy() * task.scale
                    print(f"    Manual y = (x^T w) * scale: {manual_ys}")
                    print(f"    Difference: {np.abs(cluster_ys - manual_ys)}")
                
                # Show target cluster
                print(f"\nTarget cluster (first example):")
                target_comp = data_sampler.target_components[0].item()
                target_context_ys = ys[0, target_start:target_context_end].cpu().numpy()
                target_pred_y = ys[0, predict_idx].item()
                target_context_xs = xs[0, target_start:target_context_end, :5].cpu().numpy()
                target_w = components[target_comp, :5, 0].cpu().numpy()
                target_pred_x = xs[0, predict_idx, :5].cpu().numpy()
                print(f"  Uses component {target_comp}")
                print(f"  Weight vector (first 5 dims): {target_w}")
                print(f"  Context xs (first 5 dims): {target_context_xs}")
                print(f"  Context ys: {target_context_ys}")
                print(f"  Prediction x (first 5 dims): {target_pred_x}")
                print(f"  Prediction y (to predict): {target_pred_y:.3f}")
                manual_pred = (xs[0, predict_idx:predict_idx+1] @ components[target_comp]).squeeze().item() * task.scale
                print(f"  Manual prediction = (x^T w) * scale: {manual_pred:.3f}")
                
                print(f"\nFirst example xs (first 3 points):")
                print(xs[0, :3, :5].cpu().numpy())  # First 3 points, first 5 dims
                print(f"\nFirst example ys (all points):")
                print(ys[0, :].cpu().numpy())
                print(f"ys stats: min={ys.min().item():.3f}, max={ys.max().item():.3f}, mean={ys.mean().item():.3f}, std={ys.std().item():.3f}")
                print(f"\nModel should:")
                print(f"  1. Learn components from context clusters")
                print(f"  2. Infer component {target_comp} from target cluster's context points")
                print(f"  3. Predict target point using component {target_comp}")
                print("="*60 + "\n")
        elif args.training.task == "ar_warmup" and hasattr(data_sampler, 'current_ys'):
            task = task_sampler(**task_sampler_args)
            ys = data_sampler.current_ys
        else:
            task = task_sampler(**task_sampler_args)
            ys = task.evaluate(xs)

        # Per-sequence target normalization: makes optimization independent of data scale.
        # Use normalize_ys: true when scale is large (e.g. 0.5–1.0) so the model learns structure
        # instead of "predict zero" (small scale) or failing to converge (large scale).
        if task_kwargs.get('normalize_ys', False):
            ys_mean = ys.mean(dim=1, keepdim=True)
            ys_std = ys.std(dim=1, keepdim=True) + 1e-6
            ys = (ys - ys_mean) / ys_std

        # print(f"DEBUG: xs shape: {xs.shape}, ys shape: {ys.shape}")
        # print(f"DEBUG: predict_inds: {predict_inds}")
        # print(f"DEBUG: sequence_structure total_length: {sequence_structure['total_length'] if sequence_structure else 'None'}")

        loss_func = task.get_training_metric()

        # For group_mixture_linear, optionally restrict loss to "predictable" positions only.
        # First-of-segment positions (0, C, 2C, ..., K*C) have no in-context (x,y) from the same component,
        # so the model cannot infer the mapping; loss stays ~variance there and can dominate the average.
        loss_inds = None
        if (predict_inds is not None and len(predict_inds) > 0 and
            args.training.task == "group_mixture_linear" and
            task_kwargs.get('exclude_first_of_segment_loss', False)):
            K = getattr(data_sampler, 'n_components', 2)
            C = getattr(data_sampler, 'contexts_per_component', 10)
            first_of_segment = {k * C for k in range(K + 1)}
            loss_inds = [j for j in range(len(predict_inds)) if predict_inds[j] not in first_of_segment]
            if len(loss_inds) == 0:
                loss_inds = None  # fallback to all positions

        loss, output = train_step(
            model, xs.to(device), ys.to(device), optimizer, loss_func,
            predict_inds, sequence_structure, loss_inds=loss_inds
        )

        # Why loss ~1 when predicting one position: if targets are normalized (variance 1) and predictions ~0, MSE≈1.
        if i == 0 and predict_inds is not None and len(predict_inds) == 1:
            tgt = ys[:, predict_inds[0]].to(device)
            pred = output[:, 0]
            print(f"  [single-target] pred mean={pred.mean().item():.4f} std={pred.std().item():.4f}  tgt mean={tgt.mean().item():.4f} std={tgt.std().item():.4f}  -> want pred to track tgt for loss to drop")
        
        # Critical diagnostic: check if loss is being computed correctly
        if i == 0:
            print(f"\nLOSS DEBUG (step {i}):")
            print(f"  output shape: {output.shape}")
            print(f"  output sample (first 3): {output[:3, 0].cpu().tolist() if len(output.shape) > 1 else output[:3].cpu().tolist()}")
            if predict_inds is not None and len(predict_inds) > 0:
                print(f"  ys targets shape: {ys[:, predict_inds].shape}")
                print(f"  ys targets (first 3): {ys[:3, predict_inds[0]].cpu().tolist()}")
                print(f"  Loss computed on: output vs ys[:, {predict_inds}]")
                manual_loss = ((output[:, 0] - ys[:, predict_inds[0]].to(device))**2).mean()
                print(f"  Manual MSE loss: {manual_loss.item():.6f}")
                print(f"  Reported loss: {loss:.6f}")
            else:
                print(f"  ys shape: {ys.shape}")
                print(f"  Loss computed on: output vs ys (all positions)")
            print()
        
        # Diagnostic prints for predictions
        if i == 0 or (i < 10 and i % 2 == 0):
            print(f"\nStep {i}:")
            print(f"  Loss: {loss:.4f}")
            if output is not None and len(output.shape) > 0:
                print(f"  Predictions shape: {output.shape}")
                print(f"  Predictions (first 3 examples): {output[:3, 0].cpu().tolist()}")
                print(f"  True targets (first 3 examples): {ys[:3, predict_inds[0]].cpu().tolist()}")
                pred_errors = (output[:, 0] - ys[:, predict_inds[0]].to(device)).abs()
                print(f"  Prediction errors: mean={pred_errors.mean().item():.4f}, max={pred_errors.max().item():.4f}")
                
                # For first example, show what the model should learn
                if i == 0 and args.training.task == "group_mixture_linear":
                    target_comp = data_sampler.target_components[0].item()
                    T_target = data_sampler.target_cluster_context_points
                    K = data_sampler.n_components
                    C = data_sampler.contexts_per_component
                    target_start = K * C
                    target_context_end = target_start + T_target
                    target_context_ys = ys[0, target_start:target_context_end].cpu().numpy()
                    print(f"\n  First example analysis:")
                    print(f"    Target component: {target_comp}")
                    print(f"    Target cluster context ys (for inference): {target_context_ys}")
                    print(f"    Target y (to predict): {ys[0, predict_inds[0]].item():.3f}")
                    print(f"    Model prediction: {output[0, 0].item():.3f}")
                    print(f"    Model should: infer component {target_comp} from target context, then predict")
            print()

        point_wise_loss_func = task.get_metric()
        if predict_inds is not None and len(predict_inds) > 0:
            # For multi-context: only compute loss on prediction positions
            point_wise_loss = point_wise_loss_func(output, ys[:, predict_inds].to(device)).mean(dim=0)
            point_wise_tags = predict_inds
        else:
            # Original behavior - predict all positions
            point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)
            point_wise_tags = list(range(ys.shape[1]))

        # Diagnostic: loss on first-of-segment vs rest (for group_mixture_linear)
        if (predict_inds is not None and len(predict_inds) > 0 and
            args.training.task == "group_mixture_linear" and len(point_wise_loss) == len(predict_inds)):
            K = getattr(data_sampler, 'n_components', 2)
            C = getattr(data_sampler, 'contexts_per_component', 10)
            first_of_segment = {k * C for k in range(K + 1)}
            idx_first = [j for j in range(len(predict_inds)) if predict_inds[j] in first_of_segment]
            idx_rest = [j for j in range(len(predict_inds)) if predict_inds[j] not in first_of_segment]
            loss_first_of_segment = point_wise_loss[idx_first].mean().item() if idx_first else float('nan')
            loss_rest = point_wise_loss[idx_rest].mean().item() if idx_rest else float('nan')
            if i < 3 or (i % 500 == 0 and i > 0):
                print(f"  [group_mixture] loss_first_of_segment={loss_first_of_segment:.4f} (pos {first_of_segment}), loss_rest={loss_rest:.4f}")

        if predict_inds is not None and len(predict_inds) > 0:
            # For multi-context, use a reasonable baseline
            baseline_loss = 1.0
        else:
            baseline_loss = (
                sum(
                    max(curriculum.n_dims_truncated - ii, 0)
                    for ii in range(ys.shape[1]) 
                )
                / ys.shape[1]
            )
            baseline_loss = max(baseline_loss, 0.001)
                
        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            log_data = {
                "overall_loss": loss,
                "excess_loss": loss / baseline_loss,
                "pointwise/loss": dict(
                    zip(point_wise_tags, point_wise_loss.cpu().numpy())
                ),
                "n_points": curriculum.n_points,
                "n_dims": curriculum.n_dims_truncated,
            }
            # Loss on first-of-segment vs rest (group_mixture_linear diagnostic)
            if (predict_inds is not None and len(predict_inds) > 0 and
                args.training.task == "group_mixture_linear" and len(point_wise_loss) == len(predict_inds)):
                log_data["loss_first_of_segment"] = loss_first_of_segment
                log_data["loss_rest"] = loss_rest

            # Add lag info if available
            if curriculum.lag is not None:
                log_data["lag"] = curriculum.lag

            if hasattr(task, 'get_mixture_info'):
                mix_info = task.get_mixture_info()
                log_data["n_components"] = mix_info['n_components']
                log_data["n_contexts"] = mix_info['n_contexts']
                
            if sequence_structure is not None:
                # Only log these if they exist (for backward compatibility)
                if 'context_length' in sequence_structure:
                    log_data["context_length"] = sequence_structure['context_length']
                if 'predict_length' in sequence_structure:
                    log_data["predict_length"] = sequence_structure['predict_length']

            wandb.log(log_data, step=i)

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))
            # Also save full state so you can restore to this step (e.g. copy state_9000.pt -> state.pt)
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, os.path.join(args.out_dir, f"state_{i}.pt"))

        if i % 100 == 0:
            print(f"Step {i}: Loss={loss:.4f}, Baseline={baseline_loss:.4f}, Excess={loss/baseline_loss:.4f}")
            print(f"  Curriculum: n_dims={curriculum.n_dims_truncated}, n_points={curriculum.n_points}")


def deep_merge(dict1, dict2):
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    # Inherit paths are relative to src/conf/, not the config file's directory
    if 'inherit' in config:
        base_config = {}
        config_dir = os.path.dirname(os.path.abspath(config_path))
        
        # Find src/conf base directory
        # Walk up from config file until we find 'conf' directory
        conf_base = config_dir
        while os.path.basename(conf_base) != 'conf' and conf_base != os.path.dirname(conf_base):
            conf_base = os.path.dirname(conf_base)
        
        # If we didn't find 'conf', try src/conf relative to workspace root
        if os.path.basename(conf_base) != 'conf':
            # Try to find src/conf from current working directory
            cwd = os.getcwd()
            if 'src' in cwd:
                # Extract path up to src
                parts = cwd.split(os.sep)
                src_idx = next((i for i, part in enumerate(parts) if part == 'src'), None)
                if src_idx is not None:
                    conf_base = os.sep.join(parts[:src_idx + 1] + ['conf'])
                else:
                    conf_base = os.path.join(cwd, 'src', 'conf')
            else:
                # Fallback: assume src/conf is relative to current directory
                conf_base = os.path.join(cwd, 'src', 'conf')
        
        for inherit_file in config['inherit']:
            # Try relative to conf_base first (src/conf/)
            inherit_path = os.path.join(conf_base, inherit_file)
            if not os.path.exists(inherit_path):
                # Fallback: try relative to config file directory
                inherit_path = os.path.join(config_dir, inherit_file)
            
            if not os.path.exists(inherit_path):
                # Last resort: try relative to current working directory
                cwd_conf = os.path.join(os.getcwd(), 'src', 'conf', inherit_file)
                if os.path.exists(cwd_conf):
                    inherit_path = cwd_conf
                else:
                    raise FileNotFoundError(
                        f"Could not find inherit file '{inherit_file}'. "
                        f"Tried: {os.path.join(conf_base, inherit_file)}, "
                        f"{os.path.join(config_dir, inherit_file)}, and "
                        f"{cwd_conf}"
                    )
            
            with open(inherit_path, 'r') as f:
                inherited = yaml.safe_load(f)
                base_config = deep_merge(base_config, inherited)
        
        # Remove inherit key and merge with base config
        inherit_key = config.pop('inherit', None)
        config = deep_merge(base_config, config)
    
    # Apply default values for missing keys
    defaults = {
        'training': {
            'num_tasks': None,
            'num_training_examples': None,
            'resume': True,  # if False, ignore state.pt and start from step 0
            'resume_id': None,
            'resume_extra_steps': None,  # when resuming, train this many more steps (e.g. 5000)
        },
        'wandb': {
            'entity': None,
            'name': None,
            'notes': ''
        }
    }
    
    config = deep_merge(defaults, config)
    return config


def create_args_namespace(config):
    """Convert config dict to namespace object"""
    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, Args(value))
                else:
                    setattr(self, key, value)
            
            if not hasattr(self, 'test_run'):
                setattr(self, 'test_run', False)
        
        def __repr__(self):
            return str(self.__dict__)
    
    return Args(config)


def main(args):
    if not hasattr(args, 'test_run'):
        args.test_run = False

    if not hasattr(args, 'wandb'):
        args.wandb = type('', (), {})()
    if not hasattr(args.wandb, 'project'):
        args.wandb.project = "in-context-training"
    if not hasattr(args.wandb, 'entity'):
        args.wandb.entity = None
    if not hasattr(args.wandb, 'name'):
        args.wandb.name = None
    if not hasattr(args.wandb, 'notes'):
        args.wandb.notes = ""
    if not hasattr(args.wandb, 'log_every_steps'):
        args.wandb.log_every_steps = 10

    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        try:
            wandb.init(
                dir=args.out_dir,
                project=args.wandb.project,
                entity=args.wandb.entity,
                config=args.__dict__,
                notes=args.wandb.notes,
                name=args.wandb.name,
                resume=True,
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing without wandb logging...")
            args.test_run = True

    model = build_model(args.model)
    print(args)
    print(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    print(f"Using device: {device}")

    train(model, args, device)

    if not args.test_run:
        try:
            _ = get_run_metrics(args.out_dir)  # precompute metrics for eval
        except Exception as e:
            print(f"Warning: Could not compute metrics: {e}")
            print("But training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train in-context learning model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    
    cmd_args = parser.parse_args()
    
    # Load configuration
    config = load_config(cmd_args.config)
    args = create_args_namespace(config)
    

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    main(args)
