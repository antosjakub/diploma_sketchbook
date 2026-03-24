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

base_dir = "experiments_heatwsource"  # top‑level folder
params_fixed = {
    "d": 5,
    "n_steps": 10000
}
params_gs = {
    "use_rbas": [True],
    "use_adaptive_weights": [True],
    "resampling_frequency": [750],
    "layers": ["156,156,156"],
    "n_calloc_points": [10_000],
    "bs": [512],
    "seed": [42],
}

## bool and nonbool params
## fixed in same format as params (dict)
## no need to manually slepp out args in cmd = [] - do automatically


import utility
os.makedirs(base_dir, exist_ok=True)
utility.json_dump(f"{base_dir}/grid_search_params.json", {
    "params_gs": params_gs, "params_fixed": params_fixed
})

# Build all combinations
keys = list(params_gs.keys())
values = list(params_gs.values())
arg_combos = list(itertools.product(*values))
n_combos = len(arg_combos)
print(f"Grid search consisting of {n_combos} runs.")
print(f"Saving into '{base_dir}'")


import math
n_digits = int(math.log(n_combos,10))
fmt = f"0{n_digits}d"
for i, combo in enumerate(arg_combos):
    settings = dict(zip(keys, combo))

    # Construct output_dir name from settings
    exp_id = "_".join(f"{k}={v}" for k, v in settings.items())
    t = time.strftime("%Y-%m-%d--%H:%M:%S", time.gmtime(time.time()))
    output_dir = os.path.join(base_dir, f"{t}__run{i:{fmt}}__{exp_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Build command line
    cmd = ["python", "main.py"]
    for k,v in settings.items():
        cmd.extend([f"--{k}", str(v)] if type(v) != bool else [f"--{k}"])
    for k,v in params_fixed.items():
        cmd.extend([f"--{k}", str(v)] if type(v) != bool else [f"--{k}"])
    cmd.extend(["--output_dir_name", output_dir])

    # Print for debugging
    print()
    print(f"{(i+1):{fmt}}/{n_combos:{fmt}} Running: {' '.join(cmd)}")

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
