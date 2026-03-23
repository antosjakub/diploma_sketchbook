import os
import glob
import torch
import json
import pandas as pd

experiments_dir = "experiments_big"

import utility
gsp = utility.json_load(f"{experiments_dir}/grid_search_params.json")["params"]

rows = []

for run_dir in sorted(glob.glob(os.path.join(experiments_dir, "run_*"))):
    # Read hyperparameters (you’ll need to adapt this to your config format)
    try:
        with open(os.path.join(run_dir, "model_metadata.json")) as f:
            config = json.load(f)["args"]
    except:
        continue  # or try to parse args.txt / directory name

    # Load training L2 error
    l2_path = os.path.join(run_dir, "training_l2_error.pth")
    if not os.path.exists(l2_path):
        continue

    try:
        l2_error = torch.load(l2_path, map_location="cpu")
        # Assume l2_error is a 1D tensor of shape (n_epochs,)
        if l2_error.ndim == 0:
            l2_error = l2_error.unsqueeze(0)
        if len(l2_error) < 3:
            last_3_avg = float(l2_error.mean().item())
        else:
            last_3_avg = float(l2_error[-3:].mean().item())
    except Exception as e:
        last_3_avg = float("nan")

    # Collect row
    row = {
        "last_3_l2": last_3_avg,
    }
    for k in gsp.keys():
        row[k] = config.get(k, "unknown")
        #"use_rbas": config.get("use_rbas", "unknown"),
        #"use_adaptive_weights": config.get("use_adaptive_weights", "unknown"),
        #"layers": config.get("layers", "unknown"),
        #"bs": config.get("bs", "unknown"),
        #"gamma": config.get("gamma", "unknown"),
        #"seed": config.get("seed", "unknown"),
    row["run_dir"] = run_dir
    rows.append(row)

df = pd.DataFrame(rows)
df = df.dropna()  # optional: drop failed loads
df = df.sort_values("last_3_l2", ascending=True)

print("Top 10 best runs (by avg of last 3 L₂ error):")
#print(df.head(10)[["run_dir", "last_3_l2"] + list(df.columns[1:-1])])
print(df[['last_3_l2']+list(df.columns[1:-1])])
