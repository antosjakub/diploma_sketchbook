"""Shared bootstrap/teardown helpers for main.py and main_score_pinn.py."""
import json
import os
import shutil

import torch

import loss
import utility


LOSS_KEYS = ("total", "pde", "bc", "ic", "norm")


def setup_run(args):
    """Seed, pick device, create/clear the output directory. Returns (dir_name, device)."""
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dir_name = args.output_dir
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


def make_loss_weighting(args, active_losses):
    """Build Adaptive/Constant weighting with one weight per active loss term."""
    weights = [getattr(args, f"lambda_{k}") for k in active_losses]
    if args.use_adaptive_weights:
        return loss.AdaptiveWeights(weights=torch.tensor(weights))
    return loss.ConstantWeights(weights=weights)


def make_profiler(dir_name, args):
    if not args.enable_profiler:
        return None
    return utility.Profiler(
        report_filename=f"{dir_name}/{args.profiler_report_filename}.txt",
        start_step=100,
        end_step=110,
    )


def init_losses(loss_keys=LOSS_KEYS):
    return {k: [] for k in loss_keys}


def merge_losses(dst, src):
    for k in dst:
        dst[k] += src.get(k, [])


def print_train_duration(t1, t2, label="Adam training"):
    h, m, s = utility.get_duration(t2 - t1)
    parts = [f"{label} completed in:"]
    if h > 0: parts.append(f"{h} hours")
    if m > 0: parts.append(f"{m} minutes")
    parts.append(f"{s} seconds")
    print(" ".join(parts))


def save_run(dir_name, model, losses, l2_errs, args, pde_model=None):
    """Dump model state, loss dict, l2 errors, model metadata and (optionally) pde metadata."""
    with open(f'{dir_name}/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(
            {"model_class": type(model).__name__, "args": args.__dict__},
            f,
            ensure_ascii=False,
            indent=4,
        )
    if pde_model is not None and hasattr(pde_model, "dump_pde_metadata"):
        pde_model.dump_pde_metadata(f'{dir_name}/pde_metadata.json')

    loss_name = f'{dir_name}/training_loss'
    l2_name = f'{dir_name}/training_l2_error'
    torch.save(model.state_dict(), f'{dir_name}/model.pth')
    torch.save({k: torch.tensor(v) for k, v in losses.items()}, f'{loss_name}.pth')
    torch.save(torch.tensor(l2_errs), f'{l2_name}.pth')
    print("\nResults saved.")
    return loss_name, l2_name
