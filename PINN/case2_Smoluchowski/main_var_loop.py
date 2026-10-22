

"""
PROBLEM
- since we have \tilde\rho, the IC is the hardest, TC is just 1 kinda
- move the majority of the leaning problem to the IC
- do not need to compute AD in the IC problem => training is fast


SETUP
- learn IC first
- init model weights with the prev model's ones

WORKFLOW
- separate training for the IC - just try to match the IC
    - just residual and L2 ?
    - separate args ?
    - separate script
- training loop: model -> model
    - minimization problem


ARGS
- delta t
- training args - forwards them to the main_var.py script


DIFFS

- IC training
    - general training params

- loop training
    - IC trained model to start from
    - delta t
    - minimzalization specific params
    - general training params

needs different Trainer ?
- IC trainer and Var trainer

"""
import shutil
import argparse

import os, sys
SRC_DIR = os.path.join(os.path.dirname(__file__), "../src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import run_utils
import main_runner
import main_var


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser with variational-training defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    main_runner.add_common_args(parser)
    parser.add_argument("--n_mc_steps", default=1000, type=int, help="Number of Metropolis-Hastings MC steps")
    parser.add_argument("--dt", default=0.01, type=float, help="Step size of backward Euler")
    parser.add_argument("--nt", default=3, type=int, help="Number of backward Euler step")
    parser.add_argument("--starting_dir_ic", default="run_adam_ic", type=str, help="Staring trained IC model dir path")
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
        n_steps=100,
        n_mc_steps=100,
        n_steps_decay=40,
        logging_frequency=10,
        output_dir = 'run_adam_loop',
        starting_model="run_adam_ic/model.pth",
        enable_testing=False,
        enable_profiler=False,
        enable_memory_tracking=True,
        clear_dir=True,
    )
    return parser



def main(argv: list[str] | None = None, var_step:int=0) -> None:
    parser = build_parser()
    args = run_utils.parse_args_with_config(parser, argv)

    dig_len = len(str(args.nt))
    run_utils.setup_dir(args)
    shutil.copytree(args.starting_dir_ic, f"{args.output_dir}/loop_{dig_len*'0'}", dirs_exist_ok=True)


    for i in range(args.nt):
        print("==========================================================================")
        print("==========================================================================")
        print(f"============ i = {i+1}, t = {(i+1)*args.dt} (dt={args.dt})")
        print("==========================================================================")
        print("==========================================================================")
        #python main_var.py --n_steps=50000 --gamma=0.99 --n_steps_decay=200
        main_var.main([f"--n_steps={args.n_steps}", f"--gamma={args.gamma}", f"--n_steps_decay=200",
              f"--dt={args.dt}",
              f"--starting_model={args.output_dir}/loop_{(dig_len-len(str(i)))*'0'}{i}/model.pth",
              f"--output_dir={args.output_dir}/loop_{(dig_len-len(str(i+1)))*'0'}{i+1}",
        ], i)


    # for the outer folder

    # ic and tc plot (tilde and not)
    # make an animation of \tilde\rho_n and \rho_n



if __name__ == "__main__":
    main()
