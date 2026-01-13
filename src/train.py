import os
import argparse
from random import randint
import uuid

from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from models import build_model

import wandb


torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, predict_inds=None, sequence_structure=None):
    optimizer.zero_grad()
    
    if predict_inds is None or len(predict_inds) == 0:
        output = model(xs, ys)
        loss = loss_func(output, ys)
    else:
        if sequence_structure is not None:
            output = model(xs, ys, inds=predict_inds, sequence_structure=sequence_structure)
        else:
            output = model(xs, ys, inds=predict_inds)
        loss = loss_func(output, ys[:, predict_inds])
    
    loss.backward()
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
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

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
    pbar = tqdm(range(starting_step, args.training.train_steps))

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
        elif args.training.task == "ar_warmup" and hasattr(data_sampler, 'current_ys'):
            task = task_sampler(**task_sampler_args)
            ys = data_sampler.current_ys
        else:
            task = task_sampler(**task_sampler_args)
            ys = task.evaluate(xs)

        # print(f"DEBUG: xs shape: {xs.shape}, ys shape: {ys.shape}")
        # print(f"DEBUG: predict_inds: {predict_inds}")
        # print(f"DEBUG: sequence_structure total_length: {sequence_structure['total_length'] if sequence_structure else 'None'}")

        loss_func = task.get_training_metric()

        loss, output = train_step(
            model, xs.to(device), ys.to(device), optimizer, loss_func, 
            predict_inds, sequence_structure
        )

        point_wise_loss_func = task.get_metric()
        if predict_inds is not None and len(predict_inds) > 0:
            # For multi-context: only compute loss on prediction positions
            point_wise_loss = point_wise_loss_func(output, ys[:, predict_inds].to(device)).mean(dim=0)
            point_wise_tags = predict_inds
        else:
            # Original behavior - predict all positions
            point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)
            point_wise_tags = list(range(ys.shape[1])) 

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
            
            # Add lag info if available
            if curriculum.lag is not None:
                log_data["lag"] = curriculum.lag
            
            if hasattr(task, 'get_mixture_info'):
                mix_info = task.get_mixture_info()
                log_data["n_components"] = mix_info['n_components']
                log_data["n_contexts"] = mix_info['n_contexts']
                
            if sequence_structure is not None:
                log_data["context_length"] = sequence_structure['context_length']
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
    if 'inherit' in config:
        base_config = {}
        for inherit_file in config['inherit']:
            inherit_path = os.path.join(os.path.dirname(config_path), inherit_file)
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
            'resume_id': None
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
    
    assert args.model.family in ["gpt2", "lstm"]

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
