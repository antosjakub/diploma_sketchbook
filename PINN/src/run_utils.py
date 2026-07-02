"""Shared bootstrap/teardown helpers for main.py and main_score_pinn.py."""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch

import loss
import utility


LOSS_KEYS = ("total", "pde", "bc", "ic", "norm")


def parse_args_with_config(parser, argv=None):
    """Load parser defaults from --config JSON, then parse normal CLI overrides."""
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, type=str)
    bootstrap_args, remaining_argv = bootstrap_parser.parse_known_args(argv)

    if bootstrap_args.config is not None:
        config_path = Path(bootstrap_args.config)
        with config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
            raise TypeError(f"Config file '{config_path}' must contain a JSON object.")

        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_data) - valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown config keys in '{config_path}': {unknown_keys}")

        parser.set_defaults(**config_data)

    return parser.parse_args(remaining_argv)


def save_input_config(dir_name, args):
    if getattr(args, "config", None) is None:
        return
    utility.json_dump(
        f"{dir_name}/input_config.json",
        {"config_path": args.config, "resolved_args": args.__dict__},
    )


def setup_run(args):
    """Seed, pick device, create/clear the output directory. Returns (dir_name, device)."""
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dir_name = args.output_dir
    if dir_name is None:
        raise ValueError("args.output_dir must be set before setup_run().")
    if dir_name.endswith('/'):
        dir_name = dir_name[:-1]

    if os.path.isdir(dir_name):
        print(f"Directory already exists: '{dir_name}'")
        if args.clear_dir:
            print("To the trashbin with you lot...")
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
    else:
        print(f"Creating new directory: '{dir_name}'")
        os.makedirs(dir_name)
        if args.clear_dir:
            print("Why clear the new thing me asky??")
    print()

    return dir_name, device


def make_optim(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    return optimizer, scheduler


def make_loss_weighting(args, active_losses, device=None):
    """Build Adaptive/Constant weighting with one weight per active loss term."""
    weights = torch.tensor([getattr(args, f"lambda_{k}") for k in active_losses], device=device)
    if args.use_gradnorm:
        return loss.AdaptiveWeights(weights=weights)
    return loss.ConstantWeights(weights=weights)


def make_spatial_domain(d, x_min, x_max, device=None, dtype=torch.float32):
    return torch.stack(
        [
            torch.full((d,), x_min, device=device, dtype=dtype),
            torch.full((d,), x_max, device=device, dtype=dtype),
        ],
        dim=1,
    )


def make_profiler(dir_name, args, device):
    if not args.enable_profiler:
        return None
    return utility.Profiler(
        report_filename=f"{dir_name}/{args.profiler_report_filename}",
        start_step=95,
        end_step=105,
        device=device
    )


def init_losses(loss_keys=LOSS_KEYS):
    return {k: [] for k in loss_keys}


def merge_losses(dst, src):
    for k in dst:
        dst[k] += src.get(k, [])



def save_run(dir_name, model, losses, args, device, pde_model=None, head_fn=None, loss_weighting=None, output_dim=1):
    """Dump model state, loss dict, l2 errors, model metadata and (optionally) pde metadata.

    `head_fn`: optional tag identifying the architectural head (e.g. "hardcoded_ic"),
    stored at the top level of model_metadata.json so reloaders can rebuild the model
    without string-sniffing the output directory. `None` means the default PINN head.
    `loss_weighting`: if an AdaptiveWeights instance, its `weights_history` is dumped
    to `training_loss_weights.pth` as a (n_updates, n_terms) tensor.
    """
    with open(f'{dir_name}/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                "model_class": type(model).__name__,
                "head_fn": head_fn,
                "output_dim": output_dim,
                "device": device.type,
                "args": args.__dict__,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
    if pde_model is not None and hasattr(pde_model, "dump_pde_metadata"):
        pde_model.dump_pde_metadata(f'{dir_name}/pde_metadata.json')

    loss_name = f'{dir_name}/training_loss'
    torch.save(model.state_dict(), f'{dir_name}/model.pth')
    torch.save({k: torch.tensor(v) for k, v in losses.items()}, f'{loss_name}.pth')
    if isinstance(loss_weighting, loss.AdaptiveWeights) and len(loss_weighting.weights_history) > 0:
        weights_name = f'{dir_name}/training_loss_weights'
        torch.save(torch.stack(loss_weighting.weights_history), f'{weights_name}.pth')
    print("\nResults saved.")
