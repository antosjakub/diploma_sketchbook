
entrypoint="main.py"

FIXED_PARAMS = {
    "clear_dir": False,
    "seed": 42,
    "use_hard_constrains": True,
    "active_losses": "pde,ic",
    "lambda_pde": 1.0,
    "lambda_ic": 0.1,
    ##############################################
    "d": 2,
    "layers": "64,64,64,64",
    "time_strategy": "none",
    "t_discr": "0.0, 0.5, 1.5, 3.5",
    "eps": 0.1,
    "use_lbfgs": False,
    "lambda_strategy": "fixed",
    ##############################################
    "description": "find the lambdas",

    # smthg big enough
    "n_steps": 5000,
    #"n_steps_decay": 1000,
    "logging_frequency": 10,
    # fix the dataset
    "n_res_points": 1024,
    "bs": 1024,
    "one_batch_per_epoch": True,
    "prevent_resampling": True,

    ##############################################
    "enable_profiler": False,
    "enable_memory_tracking": False,
    "enable_testing": True,
    "n_test_points": 100_000,
    "n_test_chunk_size": 100_000,
}

# Add or remove search axes here.
# Each axis value is a list. Items may be either:
# - scalars, which set one parameter with the axis name
# - dicts, for grouped parameters such as trajectory sampling settings
SEARCH_AXES = {
    #"n_steps_decay": [100, 500, 1000]
    #"ic_type": ["cauchy", "gauss"],
    #"lambda_pde": [0.1, 1.0, 10.0],
    #"lambda_ic": [0.1, 1.0, 10.0],
    #"d": [6, 4, 8],
    #"box": [
    #    {"L_min": -4.0, "L_max": 4.0},
    #    {"L_min": -6.0, "L_max": 6.0},
    #],
}


import os, sys
src_dir = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(src_dir)
import grid_search_base


if __name__ == "__main__":
    args = grid_search_base.parse_args()
    grid_search_base.run_grid_search(
        fixed_params=FIXED_PARAMS,
        search_axes=SEARCH_AXES,
        search_base_dir=os.path.dirname(__file__),
        project_dir=os.path.join(os.path.dirname(__file__), ".."),
        entrypoint=entrypoint,
        suffix=args.suffix,
        out_dir=args.out_dir,
    )
