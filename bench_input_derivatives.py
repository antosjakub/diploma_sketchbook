#!/usr/bin/env python3
"""
Benchmark input derivatives for a scalar-output neural network:

    u(X)                  : (B,)
    grad u(X)             : (B, d)
    Laplace u(X)          : (B,)
    trace(A H_u(X))       : (B,)  for constant A or per-input A(X)

This follows the PyTorch Jacobians/Hessians tutorial style:
    - torch.func.grad
    - torch.func.jacrev
    - torch.func.jacfwd
    - torch.func.hessian
    - torch.func.vmap

No stochastic trace estimators.

Example:
    python bench_input_derivatives.py --task grad --device cuda
    python bench_input_derivatives.py --task laplace --device cuda --batches 16 256 4096 10000
    python bench_input_derivatives.py --task div_const --device cuda
    python bench_input_derivatives.py --task div_x --device cuda
    python bench_input_derivatives.py --task all --device cuda
"""

import argparse
import time
from functools import partial

import torch
import torch.nn as nn
from torch.func import functional_call, grad, hessian, jacfwd, jacrev, vmap


# -------------------------
# Model and coefficient A(x)
# -------------------------

class MLP(nn.Module):
    def __init__(self, d: int, width: int = 128, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(d, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_functional_model(model):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def f_single(x):
        # x: (d,) -> scalar
        return functional_call(model, (params, buffers), (x.unsqueeze(0),)).squeeze()

    def f_batch(X):
        # X: (B, d) -> (B,)
        return functional_call(model, (params, buffers), (X,)).squeeze(-1)

    return f_single, f_batch


def A_of_x(X):
    """
    Simple input-dependent diagonal SPD-ish coefficient:
        A(x) = diag(1 + 0.1 * sin(x_j)^2)

    Returns:
        A: (B, d, d)

    This keeps div(A grad u) easy to compute exactly:
        div(A grad u)
        = trace(A H_u) + sum_j dA_jj/dx_j * du/dx_j
    """
    diag = 1.0 + 0.1 * torch.sin(X).square()
    return torch.diag_embed(diag)


def div_Ax_correction(X, grad_u):
    """
    For A(x) = diag(1 + 0.1 sin(x_j)^2),

        d A_jj / d x_j = 0.2 sin(x_j) cos(x_j)

    correction = sum_j dA_jj/dx_j * du/dx_j
    """
    d_diag = 0.2 * torch.sin(X) * torch.cos(X)
    return (d_diag * grad_u).sum(dim=-1)


# -------------------------
# Methods: gradient
# -------------------------

def grad_autograd(f_batch, X):
    X = X.detach().requires_grad_(True)
    u = f_batch(X)
    g = torch.autograd.grad(u.sum(), X, create_graph=False)[0]
    return g


def grad_vmap_func_grad(f_single, X):
    return vmap(grad(f_single))(X)


def grad_vmap_jacrev(f_single, X):
    return vmap(jacrev(f_single))(X)


# -------------------------
# Methods: Hessian contractions
# -------------------------

def lap_autograd_loop(f_batch, X):
    X = X.detach().requires_grad_(True)
    u = f_batch(X)
    g = torch.autograd.grad(u.sum(), X, create_graph=True)[0]

    lap = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
    for j in range(X.shape[1]):
        gj = g[:, j].sum()
        second_j = torch.autograd.grad(
            gj, X, create_graph=False, retain_graph=True
        )[0][:, j]
        lap = lap + second_j
    return lap


def lap_hessian(f_single, X):
    H = vmap(hessian(f_single))(X)
    return H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def lap_jacfwd_jacrev(f_single, X):
    H_fn = jacfwd(jacrev(f_single))
    H = vmap(H_fn)(X)
    return H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def lap_jacrev_jacrev(f_single, X):
    H_fn = jacrev(jacrev(f_single))
    H = vmap(H_fn)(X)
    return H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def trace_AH_hessian(f_single, X, A):
    """
    Computes trace(A H) by materializing H.

    A can be:
        (d, d)       constant A
        (B, d, d)    one A per sample
    """
    H = vmap(hessian(f_single))(X)

    if A.ndim == 2:
        return torch.einsum("ij,bij->b", A, H)
    if A.ndim == 3:
        return torch.einsum("bij,bij->b", A, H)
    raise ValueError("A must have shape (d, d) or (B, d, d)")


def trace_AH_jacfwd_jacrev(f_single, X, A):
    H_fn = jacfwd(jacrev(f_single))
    H = vmap(H_fn)(X)

    if A.ndim == 2:
        return torch.einsum("ij,bij->b", A, H)
    if A.ndim == 3:
        return torch.einsum("bij,bij->b", A, H)
    raise ValueError("A must have shape (d, d) or (B, d, d)")


def trace_AH_jacrev_jacrev(f_single, X, A):
    H_fn = jacrev(jacrev(f_single))
    H = vmap(H_fn)(X)

    if A.ndim == 2:
        return torch.einsum("ij,bij->b", A, H)
    if A.ndim == 3:
        return torch.einsum("bij,bij->b", A, H)
    raise ValueError("A must have shape (d, d) or (B, d, d)")


def div_Ax_exact_hessian(f_single, f_batch, X):
    """
    Exact div(A(x) grad u) for the A_of_x defined above:

        div(A grad u) = trace(A H) + sum_j dA_jj/dx_j * du/dx_j
    """
    A = A_of_x(X)
    trace_term = trace_AH_hessian(f_single, X, A)
    #X_req = X.detach().requires_grad_(True)
    #u = f_batch(X_req)
    #grad_u = torch.autograd.grad(u.sum(), X_req, create_graph=False)[0]
    #return trace_term + div_Ax_correction(X_req, grad_u)
    return trace_term

def div_Ax_exact_jacfwd_jacrev(f_single, f_batch, X):
    A = A_of_x(X)
    trace_term = trace_AH_jacfwd_jacrev(f_single, X, A)
    return trace_term

def div_Ax_exact_jacrev_jacrev(f_single, f_batch, X):
    A = A_of_x(X)
    trace_term = trace_AH_jacrev_jacrev(f_single, X, A)
    return trace_term


# -------------------------
# Timing utilities
# -------------------------

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_it(fn, device, warmup=5, repeat=20):
    for _ in range(warmup):
        y = fn()
    sync(device)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        y = fn()
        sync(device)
        times.append(time.perf_counter() - t0)

    t = torch.tensor(times)
    return {
        "median_ms": 1000.0 * float(t.median()),
        "mean_ms": 1000.0 * float(t.mean()),
        "min_ms": 1000.0 * float(t.min()),
        "max_ms": 1000.0 * float(t.max()),
    }


def max_abs_diff(a, b):
    return float((a.detach() - b.detach()).abs().max().cpu())


# -------------------------
# Benchmark driver
# -------------------------

def build_methods(task, f_single, f_batch, X, A_const):
    if task == "grad":
        return {
            "autograd_grad_sum": partial(grad_autograd, f_batch, X),
            "vmap_grad": partial(grad_vmap_func_grad, f_single, X),
            "vmap_jacrev": partial(grad_vmap_jacrev, f_single, X),
        }

    if task == "laplace":
        return {
            "autograd_loop": partial(lap_autograd_loop, f_batch, X),
            "vmap_hessian": partial(lap_hessian, f_single, X),
            "vmap_jacfwd_jacrev": partial(lap_jacfwd_jacrev, f_single, X),
            "vmap_jacrev_jacrev": partial(lap_jacrev_jacrev, f_single, X),
        }

    if task == "div_const":
        return {
            "vmap_hessian_trace_AH": partial(trace_AH_hessian, f_single, X, A_const),
            "vmap_jacfwd_jacrev_trace_AH": partial(trace_AH_jacfwd_jacrev, f_single, X, A_const),
            "vmap_jacrev_jacrev_trace_AH": partial(trace_AH_jacrev_jacrev, f_single, X, A_const),
        }

    if task == "div_x":
        return {
            "vmap_hessian_exact_div_Ax": partial(div_Ax_exact_hessian, f_single, f_batch, X),
            "vmap_jacfwd_jacrev_exact_div_Ax": partial(div_Ax_exact_jacfwd_jacrev, f_single, f_batch, X),
            "vmap_jacrev_jacrev_exact_div_Ax": partial(div_Ax_exact_jacrev_jacrev, f_single, f_batch, X),
        }

    raise ValueError(f"Unknown task: {task}")


def run_config(args, task, B, d, device):
    torch.manual_seed(args.seed)

    model = MLP(d=d, width=args.width, depth=args.depth).to(device=device, dtype=args.dtype)
    model.eval()

    X = torch.randn(B, d, device=device, dtype=args.dtype)

    M = torch.randn(d, d, device=device, dtype=args.dtype)
    A_const = M.T @ M / d + 0.1 * torch.eye(d, device=device, dtype=args.dtype)

    f_single, f_batch = make_functional_model(model)
    methods = build_methods(task, f_single, f_batch, X, A_const)

    print(f"\nTASK={task}  B={B}  d={d}  device={device}  width={args.width}  depth={args.depth}")
    print("-" * 92)
    print(f"{'method':40s} {'median_ms':>12s} {'mean_ms':>12} {'max_ms':>12s} {'min_ms':>12s} {'max_diff_vs_first':>18s}")
    print("-" * 92)

    reference = None

    for name, fn in methods.items():
        try:
            y = fn()
            if reference is None:
                reference = y.detach()
                diff = 0.0
            else:
                diff = max_abs_diff(y, reference)

            stats = time_it(fn, device, warmup=args.warmup, repeat=args.repeat)

            print(
                f"{name:40s} "
                f"{stats['median_ms']:12.3f} "
                f"{stats['mean_ms']:12.3f} "
                f"{stats['max_ms']:12.3f} "
                f"{stats['min_ms']:12.3f} "
                f"{diff:18.3e}"
            )

        except RuntimeError as e:
            msg = str(e).splitlines()[0]
            print(f"{name:40s} ERROR: {msg}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["grad", "laplace", "div_const", "div_x", "all"],
        default="all",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 64, 256, 1024, 4096, 16384])
    parser.add_argument("--dims", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--float64", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    args.dtype = torch.float64 if args.float64 else torch.float32

    tasks = ["grad", "laplace", "div_const", "div_x"] if args.task == "all" else [args.task]

    for task in tasks:
        for B in args.batches:
            for d in args.dims:
                run_config(args, task, B, d, device)


if __name__ == "__main__":
    main()
