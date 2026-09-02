import argparse
import torch
import matplotlib.pyplot as plt

import derivatives, architecture, utility


parser = argparse.ArgumentParser()
parser.add_argument("--d", default=4, type=int)
parser.add_argument("--layers", default="128,128,128,128", type=str)
parser.add_argument("--der_mode", default="grad", type=str, help="grad, lapl, div")
parser.add_argument("--n_calloc_pnts", default=10_000, type=int)
parser.add_argument("--n_steps", default=10, type=int)
args = parser.parse_args()


def mb(x):
    return x / 1024**2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)
if device.type != "cuda":
    raise NotADirectoryError("CUDA not available!")

D = args.d + 1
layers = utility.layers_from_string(args.layers)

model = architecture.PINN(D, layers, 1).to(device)

n_steps = args.n_steps
n_res = args.n_calloc_pnts
der_mode = args.der_mode

mem_log = []

for step in range(n_steps):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    X = torch.rand(n_res, D, dtype=torch.float32, device=device)
    X.requires_grad = True

    u = model(X)

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

    torch.cuda.synchronize()
    mem_log.append({
        "step": step,
        "allocated_mb": mb(torch.cuda.memory_allocated()),
        "reserved_mb": mb(torch.cuda.memory_reserved()),
        "peak_allocated_mb": mb(torch.cuda.max_memory_allocated()),
        "peak_reserved_mb": mb(torch.cuda.max_memory_reserved()),
    })


steps = [r["step"] for r in mem_log]

plt.figure()
plt.plot(steps, [r["allocated_mb"] for r in mem_log], label="allocated")
plt.plot(steps, [r["reserved_mb"] for r in mem_log], label="reserved")
plt.plot(steps, [r["peak_allocated_mb"] for r in mem_log], label="peak allocated")
plt.plot(steps, [r["peak_reserved_mb"] for r in mem_log], label="peak reserved")

plt.xlabel("Step")
plt.ylabel("CUDA memory [MB]")
plt.title(f"CUDA memory during {der_mode} profiling")
plt.legend()
plt.tight_layout()
plt.savefig("cuda_memory_timeseries.png", dpi=200)
plt.show()