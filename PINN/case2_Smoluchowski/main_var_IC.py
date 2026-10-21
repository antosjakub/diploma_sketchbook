"""
Train the IC step for the Smoluchowski equation.
"""

import argparse
import os
import sys
import time

import torch


SRC_DIR = os.path.join(os.path.dirname(__file__), "../src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import architecture
import main_runner
import pde_models_sde
import run_utils
import utility
from trainers import PINN_Trainer_IC


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser with variational-training defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    main_runner.add_common_args(parser)
    parser.add_argument("--n_mc_steps", default=1000, type=int, help="Number of Metropolis-Hastings MC steps")
    parser.set_defaults(
        T=2.0,
        glorot_init=True,
        act_fn="tanh",
        lr=1e-3,
        gamma=0.99,
        d=2,
        layers="64,64,64,64",
        bs=1024,
        resampling_frequency=100,
        #n_res_points=102400,
        prevent_resampling=False,
        n_steps=90,
        n_mc_steps=100,
        n_steps_decay=40,
        logging_frequency=10,
        output_dir="run_adam_ic",
        starting_model=None,
        enable_testing=False,
        enable_profiler=False,
        enable_memory_tracking=True,
        clear_dir=True,
    )
    return parser


def activation_class(name: str) -> type[torch.nn.Module]:
    """Map a configured activation name to its PyTorch module class."""
    activations = {
        "tanh": torch.nn.Tanh,
        "silu": torch.nn.SiLU,
        "softplus": torch.nn.Softplus,
        "relu": torch.nn.ReLU,
    }
    try:
        return activations[name]
    except KeyError as exc:
        choices = ", ".join(activations)
        raise ValueError(f"Unknown activation {name!r}; choose one of: {choices}.") from exc


def build_pde_model(d: int) -> pde_models_sde.SmoluchowskiVariational:
    """Construct the double-well model and its first-step variational form."""
    well_depths = torch.tensor(
        [0.8067, 1.0391, 1.0101, 0.7722, 1.0593, 1.0706, 0.9805, 0.9893, 0.9514, 1.1419]
    )
    if d > len(well_depths):
        raise ValueError(f"d={d} exceeds the {len(well_depths)} configured well depths.")

    base_model = pde_models_sde.SmoluchowskiDoubleWell(
        d=d,
        a=well_depths[:d],
        beta=1.0,
        Z=1.0,
    )
    return pde_models_sde.SmoluchowskiVariational(
        base_model,
        d=d,
        alpha=1.0,
        beta=1.0,
        dt=0.01,
    )


def summarize_losses(losses: list[float]) -> dict[str, float]:
    """Produce JSON-safe summary statistics for the final 100 loss values."""
    if not losses:
        return {}
    recent = torch.as_tensor(losses[-100:])
    return {
        "last": recent[-1].item(),
        "mean": recent.mean().item(),
        "std": recent.std(unbiased=False).item(),
        "min": recent.min().item(),
        "max": recent.max().item(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = run_utils.parse_args_with_config(parser, argv)
    if args.enable_testing:
        raise NotImplementedError(
            "TestingSuiteFP expects a space-time density model. "
            "main_var.py trains a single spatial implicit-Euler snapshot."
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_name = run_utils.setup_dir(args)
    run_utils.save_input_config(dir_name, args)

    start_time = time.time()
    pde_model = build_pde_model(args.d)
    pde_model.dump_pde_metadata(f"{dir_name}/pde_metadata.json")

    layers = utility.layers_from_string(args.layers)
    model = architecture.PINN_spatial(
        args.d,
        output_dim=1,
        layers=layers,
        activation_fn=activation_class(args.act_fn),
    ).to(device)
    if args.glorot_init:
        model.apply(architecture.init_glorot_weights)
    if args.starting_model:
        model.load_state_dict(torch.load(args.starting_model, map_location=device, weights_only=True))
        model.train()

    optimizer, scheduler = run_utils.make_optim(model, args)
    profiler = run_utils.make_profiler(dir_name, args, device)
    sampling_settings = {
        "d": args.d,
        "batch_size": args.bs,
        "n_samples": args.bs * args.resampling_frequency,
        "n_steps": args.n_mc_steps,
        "device": device,
    }

    print(f"Training PINN for IC in {args.d}D on {device}.")
    trainer = PINN_Trainer_IC(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pde_model=pde_model,
        sampling_settings=sampling_settings,
        testing_suite=None,
        profiler=profiler,
        device=device,
        dir_name=dir_name,
        grad_clip_norm=args.grad_clip_norm,
        memory_tracker=utility.MemoryTracker(device) if args.enable_memory_tracking else None,
    )
    interrupted = False
    try:
        loss_history, _, _, _ = trainer.train_adam_minibatch(
            n_steps=args.n_steps,
            n_steps_decay=args.n_steps_decay,
            resampling_frequency=args.resampling_frequency,
            logging_frequency=args.logging_frequency,
            prevent_resampling=args.prevent_resampling,
            gradnorm_update_freq=args.gradnorm_update_freq,
            crit_loss_val=args.crit_loss_val,
        )
    except KeyboardInterrupt:
        interrupted = True
        loss_history = getattr(trainer, "loss_history", [])
        print("\nTraining interrupted. Saving the current model and generating plots...")

    losses = {"variational": loss_history}
    run_utils.save_run(dir_name, model, losses, args, device, pde_model=pde_model)

    try:
        import plot_results
        import visualize_training_metrics
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("Skipping plots because matplotlib is not installed.")
    else:
        # want a single figure with 2 rows and 2 columns
        # - tilda rho IC vs tilda rho model
        # - rho IC vs rho model
        visualize_training_metrics.plot_loss(losses, f"{dir_name}/training_loss")
        args.L_min = -2.0
        args.L_max = 2.0
        plot_results.plot_variational_step(
            dir_name,
            model,
            pde_model,
            pde_model.tilde_rho_0,
            args,
            device,
        )
    report = {
        "device": device.type,
        "runtime": time.time() - start_time,
        "interrupted": interrupted,
        "loss": summarize_losses(loss_history),
    }
    if trainer.memory_history:
        report["memory_metrics"] = {
            name: max(values)
            for name, values in trainer.memory_history.items()
            if values and values[0] is not None
        }
    utility.json_dump(f"{dir_name}/report.json", report)
    status = "interrupted and saved" if interrupted else "complete"
    print(f"Training {status} in {report['runtime']:.2f}s.")


if __name__ == "__main__":
    main()
