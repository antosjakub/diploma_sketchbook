import argparse
import itertools
import os
import subprocess
import time

# What you want to vary
#params = {
#    "use_rbas": [False, True],
#    "use_adaptive_weights": [False, True],
#    "layers": ["64,64,64", "128,128,128"],
#    "bs": [256, 512],
#    "gamma": [0.9, 0.95],
#    "seed": [42, 67],
#}

base_dir = "experiments_big"  # top‑level folder
fixed = [
    "--d", "5",
    "--n_steps", "10000",
]
params = {
    "use_rbas": [True, False],
    "use_adaptive_weights": [True, False],

    "resampling_frequency": [750],
    "layers": ["128,128,128"],
    "n_calloc_points": [5000],
    "bs": [512],
    "seed": [42],
}

## bool and nonbool params
## fixed in same format as params (dict)
## no need to manually slepp out args in cmd = [] - do automatically


import utility
os.makedirs(base_dir, exist_ok=True)
utility.json_dump(f"{base_dir}/grid_search_params.json", {
    "params": params, "fixed": fixed
})

# Build all combinations
keys = list(params.keys())
values = list(params.values())
arg_combos = list(itertools.product(*values))
n_combos = len(arg_combos)
print(f"Grid search consisting of {n_combos} runs.")
print(f"Saving into '{base_dir}'")


for i, combo in enumerate(arg_combos):
    settings = dict(zip(keys, combo))

    # Construct output_dir name from settings
    exp_id = "_".join(f"{k}={v}" for k, v in settings.items())
    output_dir = os.path.join(base_dir, f"run_{i:02d}_{exp_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Build command line
    cmd = ["python", "main.py"] + fixed + [
        "--resampling_frequency", str(settings["resampling_frequency"]),
        "--layers", settings["layers"],
        "--n_calloc_points", str(settings["n_calloc_points"]),
        "--bs", str(settings["bs"]),
        "--seed", str(settings["seed"]),
        #"--gamma", str(settings["gamma"]),
    ]
    if settings["use_rbas"]:
        cmd.append("--use_rbas")
    if settings["use_adaptive_weights"]:
        cmd.append("--use_adaptive_weights")
    cmd.extend(["--output_dir_name", output_dir])

    # Print for debugging
    print()
    print(f"{(i+1):02d}/{n_combos} Running: {' '.join(cmd)}")

    # Timing and log capture
    start_time = time.time()
    log_file = os.path.join(output_dir, "stdout_stderr.log")

    with open(log_file, "w", buffering=1) as log_fp:
        proc = subprocess.run(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
    end_time = time.time()

    # Save time in seconds
    time_file = os.path.join(output_dir, "time.txt")
    with open(time_file, "w") as f:
        f.write(f"{end_time - start_time:.3f}\n")

    if proc.returncode != 0:
        print(f"Run {i} FAILED (code {proc.returncode})")
    else:
        print(f"Run {i} OK, time: {end_time - start_time:.1f}s")
