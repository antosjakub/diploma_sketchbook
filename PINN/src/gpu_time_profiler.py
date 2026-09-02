import argparse
import torch
import matplotlib.pyplot as plt

import derivatives, architecture, utility


parser = argparse.ArgumentParser()
parser.add_argument("--d", default=4, type=int)
parser.add_argument("--der_mode", default="grad", type=str, help="grad, lapl, div")
parser.add_argument("--layers", default="128,128,128,128", type=str)
parser.add_argument("--n_calloc_pnts", default=10_000, type=int)
parser.add_argument("--n_steps", default=100, type=int)
parser.add_argument("--n_steps_warmup", default=10, type=int)
parser.add_argument("--out_dir", default="out_dir", type=str)
args = parser.parse_args()

#out_dir = f"{args.out_dir}/d={d},der_mode={der_mode},layer={layers},n_calloc_pnts={n_calloc_pnts},n_steps={n_steps}"
#out_dir = f"out_dir"
out_dir = args.out_dir
import os
os.makedirs(out_dir, exist_ok=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)

D = args.d + 1
layers = utility.layers_from_string(args.layers)

model = architecture.PINN(D, layers, 1).to(device)

n_steps = args.n_steps
warmup = args.n_steps_warmup
n_res = args.n_calloc_pnts
der_mode = args.der_mode

time_log = []

import time
for step in range(n_steps):
    t0 = time.time()
    X = torch.rand(n_res, D, dtype=torch.float32, device=device)
    X.requires_grad = True
    t1 = time.time()
    t_sample = t1-t0

    t0 = time.time()
    u = model(X)
    t1 = time.time()
    t_forward = t1-t0

    t0 = time.time()
    if der_mode == "grad":
        u_grad = derivatives.compute_grad(X, u, torch.ones_like(u))

    elif der_mode == "lapl":
        u, grad_u, spatial_laplace_u = derivatives.compute_derivatives(model, X)

    elif der_mode == "div":
        u_grad = derivatives.compute_grad(X, u, torch.ones_like(u))
        u_t = u_grad[:, -1:]
        u_jac = u_grad[:, :-1]
        model_u_grad = lambda X_in: u_jac
        score, jac = derivatives.compute_score_and_jacobian(model_u_grad, X)

    else:
        raise ValueError(f"Unknown der_mode: {der_mode}")
    t1 = time.time()
    t_der_comp = t1-t0

    time_log.append({
        "step": step,
        "t_sample": t_sample,
        "t_forward": t_forward,
        "t_der_comp": t_der_comp,
    })

summary = {
    "args": vars(args),
    "mean": {
        "t_sample":   sum(r["t_sample"]   for r in time_log[warmup:])/(n_steps-warmup),
        "t_forward":  sum(r["t_forward"]  for r in time_log[warmup:])/(n_steps-warmup),
        "t_der_comp": sum(r["t_der_comp"] for r in time_log[warmup:])/(n_steps-warmup),
    }
}
import json
with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)


import csv
with open(f"{out_dir}/timeseries_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "step",
            "t_sample",
            "t_forward",
            "t_der_comp",
        ],
    )
    writer.writeheader()
    writer.writerows(time_log)

print("All saved")

steps = [r["step"] for r in time_log]

for type in ["full", "warmup"]:
    if type == "full":
        offset = 0
    elif type == "warmup":
        offset = warmup

    plt.figure()
    plt.plot(steps[offset:], [r["t_sample"]   for r in time_log[offset:]], label="t_sample")
    plt.plot(steps[offset:], [r["t_forward"]  for r in time_log[offset:]], label="t_forward")
    plt.plot(steps[offset:], [r["t_der_comp"] for r in time_log[offset:]], label="t_der_comp")

    plt.xlabel("Step")
    plt.ylabel("CUDA time")
    plt.title(f"CUDA time during {der_mode} profiling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/timeseries_plot_{type}.png", dpi=200)
    plt.show()